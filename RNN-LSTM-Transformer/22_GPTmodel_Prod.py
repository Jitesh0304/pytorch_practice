"""
    data
     ↓
    BPE tokenizer
     ↓
    token IDs
     ↓
    dataset / batches
     ↓
    token embedding
     ↓
    RoPE
     ↓
    GQA
     ↓
    RMSNorm
     ↓
    SwiGLU
     ↓
    Transformer blocks
     ↓
    LM Head
     ↓
    Cross Entropy
     ↓
    AdamW
     ↓
    training
     ↓
    KV-cache generation
     ↓
    BPE decoding
"""


import math
import re
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

with open("./data.txt", "r", encoding="utf-8") as f:
    text = f.read()

text = text.strip()

print("Characters:", len(text))
print(text[:500])

# random.seed(42)

words = re.findall(r"\S+", text.lower())

# random.shuffle(words)

split = int(0.8 * len(words))

train_words = words[:split]
val_words = words[split:]

print("Training words:", len(train_words))
print("Validation words:", len(val_words))


# BPE tokenizer (Byte Pair Encoding)

def word_to_symbols(word):
    return list("▁" + word)


def count_pairs(word_sequences):

    pair_counts = Counter()

    for symbols, frequency in word_sequences:

        for i in range(len(symbols) - 1):

            pair = (
                symbols[i],
                symbols[i + 1]
            )

            pair_counts[pair] += frequency

    return pair_counts


def merge_pair(symbols, pair):

    result = []

    i = 0

    while i < len(symbols):

        if (
            i < len(symbols) - 1
            and
            (symbols[i], symbols[i + 1]) == pair
        ):

            result.append(
                symbols[i] + symbols[i + 1]
            )

            i += 2

        else:

            result.append(symbols[i])

            i += 1

    return result


def train_bpe(words, num_merges):

    frequencies = Counter(words)

    word_sequences = [
        (word_to_symbols(word), freq)
        for word, freq in frequencies.items()
    ]

    vocab = set()

    for symbols, _ in word_sequences:
        vocab.update(symbols)

    merges = []

    for step in range(num_merges):

        pair_counts = count_pairs(
            word_sequences
        )

        if not pair_counts:
            break

        best_pair, count = (
            pair_counts.most_common(1)[0]
        )

        merges.append(best_pair)

        new_token = (
            best_pair[0]
            +
            best_pair[1]
        )

        vocab.add(new_token)

        word_sequences = [
            (
                merge_pair(symbols, best_pair),
                freq
            )
            for symbols, freq
            in word_sequences
        ]

        if step % 10 == 0:

            print(
                f"BPE merge {step}: "
                f"{best_pair} "
                f"count={count}"
            )

    return vocab, merges


vocab, merges = train_bpe(
    train_words,
    num_merges=200
)

print("BPE vocabulary:", len(vocab))
print("Number of merges:", len(merges))

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
}


class BPETokenizer:

    def __init__(
        self,
        vocab,
        merges
    ):

        self.merges = merges

        self.special_tokens = SPECIAL_TOKENS

        self.token_to_id = dict(
            SPECIAL_TOKENS
        )

        next_id = len(
            self.token_to_id
        )

        for token in sorted(vocab):

            if token not in self.token_to_id:

                self.token_to_id[token] = next_id

                next_id += 1

        self.id_to_token = {
            idx: token
            for token, idx
            in self.token_to_id.items()
        }

        self.merge_ranks = {
            pair: rank
            for rank, pair
            in enumerate(self.merges)
        }


    def encode_word(self, word):

        symbols = word_to_symbols(
            word
        )

        for pair in self.merges:

            symbols = merge_pair(
                symbols,
                pair
            )

        return symbols


    def encode(
        self,
        text,
        add_bos=False,
        add_eos=False
    ):

        words = re.findall(
            r"\S+",
            text.lower()
        )

        tokens = []

        if add_bos:
            tokens.append("<BOS>")

        for word in words:

            symbols = self.encode_word(
                word
            )

            tokens.extend(symbols)

        if add_eos:
            tokens.append("<EOS>")

        ids = []

        unk_id = self.token_to_id[
            "<UNK>"
        ]

        for token in tokens:

            ids.append(
                self.token_to_id.get(
                    token,
                    unk_id
                )
            )

        return ids


    def decode(self, ids):

        pieces = []

        for idx in ids:

            token = self.id_to_token.get(
                int(idx),
                "<UNK>"
            )

            if token in {
                "<PAD>",
                "<BOS>",
                "<EOS>"
            }:
                continue

            pieces.append(token)

        text = "".join(pieces)

        return text.replace(
            "▁",
            " "
        ).strip()


tokenizer = BPETokenizer(
    vocab,
    merges
)

print(
    "Tokenizer vocabulary:",
    len(tokenizer.token_to_id)
)

# sample = "the transformer is powerful"
#
# ids = tokenizer.encode(sample)
#
# print("IDs:")
# print(ids)
#
# decoded = tokenizer.decode(ids)
#
# print("Decoded:")
# print(decoded)

train_ids = tokenizer.encode(
    " ".join(train_words)
)

val_ids = tokenizer.encode(
    " ".join(val_words)
)

print(
    "Train tokens:",
    len(train_ids)
)

print(
    "Validation tokens:",
    len(val_ids)
)

train_data = torch.tensor(
    train_ids,
    dtype=torch.long
)

print("train data size ", train_data.shape)

val_data = torch.tensor(
    val_ids,
    dtype=torch.long
)



def get_batch(
    data,
    batch_size,
    block_size,
    device
):
    max_start = (
        len(data)
        - block_size
        - 1
    )

    # max_start = len(data) - block_size

    if max_start <= 0:
        raise ValueError(
            f"Dataset is too small: "
            f"len(data)={len(data)}, "
            f"block_size={block_size}. "
            f"Need at least block_size + 1 tokens."
        )

    starts = torch.randint(
        0,
        max_start,
        (batch_size,)
    )

    x = torch.stack([
        data[
            start:
            start + block_size
        ]
        for start in starts
    ])

    y = torch.stack([
        data[
            start + 1:
            start + block_size + 1
        ]
        for start in starts
    ])

    return (
        x.to(device),
        y.to(device)
    )



class RMSNorm(nn.Module):

    def __init__(
        self,
        dim,
        eps=1e-6
    ):

        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(dim)
        )

    def forward(self, x):

        rms = torch.sqrt(
            x.pow(2).mean(
                dim=-1,
                keepdim=True
            )
            + self.eps
        )

        return (
            x / rms
        ) * self.weight


class RoPE(nn.Module):

    def __init__(
        self,
        head_dim,
        max_seq_len,
        base=10000
    ):

        super().__init__()

        assert head_dim % 2 == 0

        inv_freq = 1.0 / (
            base ** (
                torch.arange(
                    0,
                    head_dim,
                    2
                ).float()
                / head_dim
            )
        )

        positions = torch.arange(
            max_seq_len
        ).float()

        freqs = torch.outer(
            positions,
            inv_freq
        )

        self.register_buffer(
            "cos",
            freqs.cos(),
            persistent=False
        )

        self.register_buffer(
            "sin",
            freqs.sin(),
            persistent=False
        )

    def forward(
        self,
        q,
        k,
        start_pos=0
    ):

        T = q.size(-2)

        cos = self.cos[
            start_pos:
            start_pos + T
        ]

        sin = self.sin[
            start_pos:
            start_pos + T
        ]

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_even = q[..., 0::2]
        q_odd = q[..., 1::2]

        k_even = k[..., 0::2]
        k_odd = k[..., 1::2]

        q = torch.stack(
            [
                q_even * cos - q_odd * sin,
                q_even * sin + q_odd * cos
            ],
            dim=-1
        ).flatten(-2)

        k = torch.stack(
            [
                k_even * cos - k_odd * sin,
                k_even * sin + k_odd * cos
            ],
            dim=-1
        ).flatten(-2)

        return q, k


class GQAAttention(nn.Module):

    def __init__(
        self,
        d_model,
        num_q_heads,
        num_kv_heads,
        max_seq_len
    ):

        super().__init__()

        assert d_model % num_q_heads == 0

        assert (
            num_q_heads % num_kv_heads == 0
        )

        self.d_model = d_model

        self.num_q_heads = num_q_heads

        self.num_kv_heads = num_kv_heads

        self.head_dim = (
            d_model // num_q_heads
        )

        self.group_size = (
            num_q_heads // num_kv_heads
        )

        self.q_proj = nn.Linear(
            d_model,
            num_q_heads * self.head_dim,
            bias=False
        )

        self.k_proj = nn.Linear(
            d_model,
            num_kv_heads * self.head_dim,
            bias=False
        )

        self.v_proj = nn.Linear(
            d_model,
            num_kv_heads * self.head_dim,
            bias=False
        )

        self.o_proj = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.rope = RoPE(
            self.head_dim,
            max_seq_len
        )

    def init_cache(
        self,
        batch_size,
        max_seq_len,
        device,
        dtype
    ):

        k_cache = torch.zeros(
            batch_size,
            self.num_kv_heads,
            max_seq_len,
            self.head_dim,
            device=device,
            dtype=dtype
        )

        v_cache = torch.zeros_like(
            k_cache
        )

        return k_cache, v_cache


    def forward(
        self,
        x,
        k_cache=None,
        v_cache=None,
        start_pos=0,
        use_cache=False
    ):

        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            B,
            T,
            self.num_q_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.num_kv_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.num_kv_heads,
            self.head_dim
        ).transpose(1, 2)

        q, k = self.rope(
            q,
            k,
            start_pos
        )


        if use_cache:

            end_pos = start_pos + T

            k_cache[
                :,
                :,
                start_pos:end_pos,
                :
            ] = k

            v_cache[
                :,
                :,
                start_pos:end_pos,
                :
            ] = v

            k_all = k_cache[
                :,
                :,
                :end_pos,
                :
            ]

            v_all = v_cache[
                :,
                :,
                :end_pos,
                :
            ]

        else:

            k_all = k
            v_all = v

        q_grouped = q.view(
            B,
            self.num_kv_heads,
            self.group_size,
            T,
            self.head_dim
        )

        scores = torch.einsum(
            "b h g t d, b h s d -> b h g t s",
            q_grouped,
            k_all
        )

        scores = scores / math.sqrt(
            self.head_dim
        )

        if not use_cache:

            causal_mask = torch.tril(
                torch.ones(
                    T,
                    T,
                    device=x.device,
                    dtype=torch.bool
                )
            )

            scores = scores.masked_fill(
                ~causal_mask,
                torch.finfo(
                    scores.dtype
                ).min
            )

        weights = F.softmax(
            scores,
            dim=-1
        )

        output = torch.einsum(
            "b h g t s, b h s d -> b h g t d",
            weights,
            v_all
        )

        output = output.reshape(
            B,
            self.num_q_heads,
            T,
            self.head_dim
        )

        output = output.transpose(
            1,
            2
        )

        output = output.contiguous().view(
            B,
            T,
            self.d_model
        )

        output = self.o_proj(
            output
        )

        return (
            output,
            k_cache,
            v_cache
        )


class SwiGLU(nn.Module):

    def __init__(
        self,
        d_model,
        ffn_dim
    ):

        super().__init__()

        self.gate_proj = nn.Linear(
            d_model,
            ffn_dim,
            bias=False
        )

        self.up_proj = nn.Linear(
            d_model,
            ffn_dim,
            bias=False
        )

        self.down_proj = nn.Linear(
            ffn_dim,
            d_model,
            bias=False
        )

    def forward(self, x):

        gate = F.silu(
            self.gate_proj(x)
        )

        value = self.up_proj(x)

        return self.down_proj(
            gate * value
        )


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model,
        num_q_heads,
        num_kv_heads,
        ffn_dim,
        max_seq_len
    ):

        super().__init__()

        self.norm1 = RMSNorm(
            d_model
        )

        self.attention = GQAAttention(
            d_model,
            num_q_heads,
            num_kv_heads,
            max_seq_len
        )

        self.norm2 = RMSNorm(
            d_model
        )

        self.ffn = SwiGLU(
            d_model,
            ffn_dim
        )

    def forward(
        self,
        x,
        k_cache=None,
        v_cache=None,
        start_pos=0,
        use_cache=False
    ):

        attention_output, k_cache, v_cache = (
            self.attention(
                self.norm1(x),
                k_cache,
                v_cache,
                start_pos,
                use_cache
            )
        )

        x = x + attention_output

        x = x + self.ffn(
            self.norm2(x)
        )

        return (
            x,
            k_cache,
            v_cache
        )


class GPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_layers=4,
        num_q_heads=8,
        num_kv_heads=2,
        ffn_dim=256,
        max_seq_len=64
    ):

        super().__init__()

        self.vocab_size = vocab_size

        self.d_model = d_model

        self.max_seq_len = max_seq_len

        self.num_layers = num_layers

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_q_heads,
                num_kv_heads,
                ffn_dim,
                max_seq_len
            )
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(
            d_model
        )

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False
        )

        # Weight tying
        self.lm_head.weight = (
            self.token_embedding.weight
        )

        self.apply(
            self._init_weights
        )


    def _init_weights(self, module):

        if isinstance(
            module,
            nn.Linear
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )


    def forward(
        self,
        input_ids,
        caches=None,
        start_pos=0,
        use_cache=False
    ):

        B, T = input_ids.shape

        if start_pos + T > self.max_seq_len:

            raise ValueError(
                "Sequence exceeds max_seq_len"
            )

        x = self.token_embedding(
            input_ids
        )

        new_caches = []

        for layer_idx, block in enumerate(
            self.blocks
        ):

            if caches is not None:

                k_cache, v_cache = (
                    caches[layer_idx]
                )

            else:

                k_cache = None
                v_cache = None

            x, k_cache, v_cache = block(
                x,
                k_cache,
                v_cache,
                start_pos,
                use_cache
            )

            if use_cache:

                new_caches.append(
                    (k_cache, v_cache)
                )

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return (
            logits,
            new_caches if use_cache else None
        )


batch_size = 8
block_size = 32

print("Train tokens:", len(train_ids))
print("Validation tokens:", len(val_ids))

if len(val_ids) < block_size + 1:
    raise ValueError(
        f"Validation set has only {len(val_ids)} tokens, "
        f"but block_size={block_size} requires at least "
        f"{block_size + 1} tokens."
    )

x, y = get_batch(
    train_data,
    batch_size,
    block_size,
    device
)

print(x.shape)
print(y.shape)

vocab_size = len(
    tokenizer.token_to_id
)

print(
    "Vocabulary size:",
    vocab_size
)

model = GPT(
    vocab_size=vocab_size,
    d_model=128,
    num_layers=4,
    num_q_heads=8,
    num_kv_heads=2,
    ffn_dim=256,
    max_seq_len=64
).to(device)


num_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Parameters: {num_parameters:,}"
)

#
# x, y = get_batch(
#     train_data,
#     batch_size,
#     block_size,
#     device
# )
#
logits, _ = model(x)

print("Input:", x.shape)
print("Logits:", logits.shape)
print("Target:", y.shape)


loss = F.cross_entropy(
    logits.reshape(
        -1,
        vocab_size
    ),
    y.reshape(-1)
)

print("Initial loss:", loss.item())


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01,
    betas=(0.9, 0.95)
)


max_steps = 5000
warmup_steps = 200
max_lr = 3e-4
min_lr = 3e-5


def get_lr(step):

    if step < warmup_steps:

        return max_lr * (
            step + 1
        ) / warmup_steps

    if step >= max_steps:

        return min_lr

    progress = (
        step - warmup_steps
    ) / (
        max_steps - warmup_steps
    )

    cosine = 0.5 * (
        1
        +
        math.cos(
            math.pi * progress
        )
    )

    return (
        min_lr
        +
        cosine * (
            max_lr - min_lr
        )
    )


@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for name, data in [
        ("train", train_data),
        ("val", val_data)
    ]:

        losses = []

        for _ in range(20):

            x, y = get_batch(
                data,
                batch_size,
                block_size,
                device
            )

            logits, _ = model(x)

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    vocab_size
                ),
                y.reshape(-1)
            )

            losses.append(
                loss.item()
            )

        results[name] = (
            sum(losses)
            /
            len(losses)
        )

    model.train()

    return results



for step in range(max_steps):

    lr = get_lr(step)

    for param_group in optimizer.param_groups:

        param_group["lr"] = lr

    x, y = get_batch(
        train_data,
        batch_size,
        block_size,
        device
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    logits, _ = model(x)

    loss = F.cross_entropy(
        logits.reshape(
            -1,
            vocab_size
        ),
        y.reshape(-1)
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    if step % 100 == 0:

        losses = estimate_loss()

        print(
            f"step {step:5d} | "
            f"lr {lr:.6f} | "
            f"train {losses['train']:.4f} | "
            f"val {losses['val']:.4f}"
        )


@torch.no_grad()
def create_kv_cache(
    model,
    batch_size,
    device
):

    caches = []

    for block in model.blocks:

        k_cache, v_cache = (
            block.attention.init_cache(
                batch_size,
                model.max_seq_len,
                device,
                model.token_embedding.weight.dtype
            )
        )

        caches.append(
            (k_cache, v_cache)
        )

    return caches


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50
):

    model.eval()

    prompt_ids = tokenizer.encode(
        prompt
    )

    if len(prompt_ids) == 0:

        raise ValueError(
            "Prompt produced no tokens"
        )

    if len(prompt_ids) >= model.max_seq_len:

        prompt_ids = prompt_ids[
            -model.max_seq_len:
        ]

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device
    )

    caches = create_kv_cache(
        model,
        batch_size=1,
        device=device
    )

    # Process entire prompt first
    logits, caches = model(
        input_ids,
        caches=caches,
        start_pos=0,
        use_cache=True
    )

    generated = input_ids

    for step in range(max_new_tokens):

        next_logits = logits[:, -1, :]

        if temperature <= 0:

            next_token = torch.argmax(
                next_logits,
                dim=-1,
                keepdim=True
            )

        else:

            next_logits = (
                next_logits
                /
                temperature
            )

            if top_k is not None:

                k = min(
                    top_k,
                    next_logits.size(-1)
                )

                values, _ = torch.topk(
                    next_logits,
                    k
                )

                threshold = values[
                    :, -1
                ].unsqueeze(-1)

                next_logits = torch.where(
                    next_logits < threshold,
                    torch.full_like(
                        next_logits,
                        float("-inf")
                    ),
                    next_logits
                )

            probs = F.softmax(
                next_logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

        generated = torch.cat(
            [
                generated,
                next_token
            ],
            dim=1
        )

        current_pos = (
            generated.size(1) - 1
        )

        if current_pos >= model.max_seq_len:

            break

        logits, caches = model(
            next_token,
            caches=caches,
            start_pos=current_pos,
            use_cache=True
        )

    return tokenizer.decode(
        generated[0].tolist()
    )


prompt = "the transformer"

output = generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=30,
    temperature=0.8,
    top_k=20
)

print(output)


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "token_to_id": tokenizer.token_to_id,
        "merges": tokenizer.merges,
    },
    "gpt_checkpoint.pt"
)

"""
                 INPUT TEXT
                     │
                     ▼
              ┌─────────────┐
              │     BPE     │
              └──────┬──────┘
                     │
                     ▼
              TOKEN IDs
                     │
                     ▼
           ┌──────────────────┐
           │ Token Embedding  │
           └────────┬─────────┘
                    │
                    ▼
          ┌───────────────────────┐
          │ Transformer Block × 4 │
          │                       │
          │ RMSNorm               │
          │    ↓                  │
          │ Q/K/V projections     │
          │    ↓                  │
          │ RoPE                  │
          │    ↓                  │
          │ GQA                   │
          │    ↓                  │
          │ Causal Mask           │
          │    ↓                  │
          │ Attention             │
          │    ↓                  │
          │ Residual              │
          │                       │
          │ RMSNorm               │
          │    ↓                  │
          │ SwiGLU                │
          │    ↓                  │
          │ Residual              │
          └───────────┬───────────┘
                      │
                      ▼
                  RMSNorm
                      │
                      ▼
                  LM Head
                      │
                      ▼
                   LOGITS
                      │
                      ▼
                Cross Entropy
                      │
                      ▼
                  BACKWARD
                      │
                      ▼
                   AdamW
                      │
                      ▼
                UPDATED MODEL

And generation:

                 PROMPT
                    │
                    ▼
                   BPE
                    │
                    ▼
                Token IDs
                    │
                    ▼
                  GPT
                    │
                    ▼
               KV Cache
                    │
                    ▼
              Next-token logits
                    │
                    ▼
              Temperature
                    │
                    ▼
                  Top-K
                    │
                    ▼
                 Sample
                    │
                    ▼
              New token
                    │
                    └──────────┐
                               │
                               ▼
                          KV Cache
                               │
                               ▼
                         Next token
                               │
                              ...
                               │
                               ▼
                            BPE
                               │
                               ▼
                          GENERATED TEXT
"""




