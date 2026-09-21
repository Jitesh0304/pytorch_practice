import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter



device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(device)


class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model,
        eps=1e-6
    ):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(d_model)
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


class RotaryEmbedding(nn.Module):

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
        )

        freqs = torch.outer(
            positions,
            inv_freq
        )

        self.register_buffer(
            "cos",
            freqs.cos()
        )

        self.register_buffer(
            "sin",
            freqs.sin()
        )

    def rotate(
        self,
        x,
        cos,
        sin
    ):

        x_even = x[..., 0::2]

        x_odd = x[..., 1::2]

        rotated_even = (
            x_even * cos
            -
            x_odd * sin
        )

        rotated_odd = (
            x_even * sin
            +
            x_odd * cos
        )

        x = torch.stack(
            [
                rotated_even,
                rotated_odd
            ],
            dim=-1
        )

        return x.flatten(-2)

    def forward(
        self,
        q,
        k,
        position_offset=0
    ):

        T = q.shape[-2]

        cos = self.cos[
            position_offset:
            position_offset + T
        ]

        sin = self.sin[
            position_offset:
            position_offset + T
        ]

        q = self.rotate(
            q,
            cos,
            sin
        )

        k = self.rotate(
            k,
            cos,
            sin
        )

        return q, k


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

        self.rope = RotaryEmbedding(
            self.head_dim,
            max_seq_len
        )


    def forward(
        self,
        x,
        past_k=None,
        past_v=None
    ):

        B, T, _ = x.shape

        # --------------------
        # Project
        # --------------------

        q = self.q_proj(x)

        k = self.k_proj(x)

        v = self.v_proj(x)

        # --------------------
        # Reshape Q
        # --------------------

        q = q.view(
            B,
            T,
            self.num_q_heads,
            self.head_dim
        )

        q = q.transpose(1, 2)

        # --------------------
        # Reshape K
        # --------------------

        k = k.view(
            B,
            T,
            self.num_kv_heads,
            self.head_dim
        )

        k = k.transpose(1, 2)

        # --------------------
        # Reshape V
        # --------------------

        v = v.view(
            B,
            T,
            self.num_kv_heads,
            self.head_dim
        )

        v = v.transpose(1, 2)

        if past_k is not None:

            position_offset = (
                past_k.shape[2]
            )

        else:

            position_offset = 0

        q, k = self.rope(
            q,
            k,
            position_offset
        )

        if past_k is not None:

            k = torch.cat(
                [past_k, k],
                dim=2
            )

        if past_v is not None:

            v = torch.cat(
                [past_v, v],
                dim=2
            )

        k_attn = k.repeat_interleave(
            self.group_size,
            dim=1
        )

        v_attn = v.repeat_interleave(
            self.group_size,
            dim=1
        )

        scores = (
            q @ k_attn.transpose(
                -2,
                -1
            )
        )

        scores = scores / math.sqrt(
            self.head_dim
        )


        if past_k is None:

            mask = torch.tril(
                torch.ones(
                    T,
                    T,
                    device=x.device,
                    dtype=torch.bool
                )
            )

            scores = scores.masked_fill(
                ~mask,
                float("-inf")
            )

        weights = F.softmax(
            scores,
            dim=-1
        )

        output = weights @ v_attn

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

        return output, k, v


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
        past_k=None,
        past_v=None
    ):

        # Attention
        norm_x = self.norm1(x)

        attention_output, k, v = (
            self.attention(
                norm_x,
                past_k,
                past_v
            )
        )

        x = x + attention_output

        # Feed-forward
        ffn_output = self.ffn(
            self.norm2(x)
        )

        x = x + ffn_output

        return x, k, v


class GPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        num_layers,
        num_q_heads,
        num_kv_heads,
        ffn_dim,
        max_seq_len
    ):
        super().__init__()

        self.vocab_size = vocab_size

        self.d_model = d_model

        self.max_seq_len = max_seq_len

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

    def forward(
        self,
        input_ids,
        past_kv=None
    ):

        B, T = input_ids.shape

        assert (
            T <= self.max_seq_len
        )

        x = self.token_embedding(
            input_ids
        )

        new_kv = []

        for i, block in enumerate(
            self.blocks
        ):

            if past_kv is not None:

                past_k = past_kv[i][0]

                past_v = past_kv[i][1]

            else:

                past_k = None
                past_v = None

            x, k, v = block(
                x,
                past_k,
                past_v
            )

            new_kv.append(
                (k, v)
            )

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits, new_kv


def initialize_words(words):

    word_tokens = {}

    for word in words:

        word_tokens[word] = list(word)

    return word_tokens


def get_initial_vocab(words):

    vocab = set()

    for word in words:

        for char in word:

            vocab.add(char)

    return vocab


def count_pairs(
    word_tokens
):

    pair_counts = Counter()

    for tokens in word_tokens.values():

        for i in range(
            len(tokens) - 1
        ):

            pair = (
                tokens[i],
                tokens[i + 1]
            )

            pair_counts[pair] += 1

    return pair_counts


def get_best_pair(
    pair_counts
):

    if not pair_counts:
        return None

    return pair_counts.most_common(1)[0][0]


def merge_pair(
    tokens,
    pair
):

    new_tokens = []

    i = 0

    while i < len(tokens):

        if (
            i < len(tokens) - 1
            and
            (tokens[i], tokens[i + 1])
            == pair
        ):

            merged = (
                tokens[i]
                +
                tokens[i + 1]
            )

            new_tokens.append(
                merged
            )

            i += 2

        else:

            new_tokens.append(
                tokens[i]
            )

            i += 1

    return new_tokens


def merge_all_words(
    word_tokens,
    pair
):

    for word in word_tokens:

        word_tokens[word] = merge_pair(
            word_tokens[word],
            pair
        )


def train_bpe(
    words,
    num_merges
):

    word_tokens = initialize_words(
        words
    )

    vocab = get_initial_vocab(
        words
    )

    merges = []

    for _ in range(num_merges):

        pair_counts = count_pairs(
            word_tokens
        )

        if not pair_counts:
            break

        best_pair = get_best_pair(
            pair_counts
        )

        merges.append(
            best_pair
        )

        new_token = (
            best_pair[0]
            +
            best_pair[1]
        )

        vocab.add(
            new_token
        )

        merge_all_words(
            word_tokens,
            best_pair
        )

    return vocab, merges


class BPETokenizer:

    def __init__(
        self,
        vocab,
        merges
    ):

        self.merges = merges

        self.vocab = vocab

        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3
        }

        self.token_to_id = dict(
            self.special_tokens
        )

        next_id = len(
            self.token_to_id
        )

        for token in sorted(vocab):

            if token not in self.token_to_id:

                self.token_to_id[
                    token
                ] = next_id

                next_id += 1

        self.id_to_token = {
            idx: token
            for token, idx
            in self.token_to_id.items()
        }

    def encode(
        self,
        text,
        add_bos=False,
        add_eos=False
    ):

        text = text.replace(
            " ",
            "▁"
        )

        tokens = list(text)

        for pair in self.merges:

            tokens = merge_pair(
                tokens,
                pair
            )

        ids = []

        if add_bos:

            ids.append(
                self.token_to_id["<BOS>"]
            )

        for token in tokens:

            ids.append(
                self.token_to_id.get(
                    token,
                    self.token_to_id["<UNK>"]
                )
            )

        if add_eos:

            ids.append(
                self.token_to_id["<EOS>"]
            )

        return ids


    def decode(
        self,
        ids
    ):

        tokens = []

        for idx in ids:

            token = self.id_to_token.get(
                idx,
                "<UNK>"
            )

            if token in [
                "<PAD>",
                "<BOS>",
                "<EOS>"
            ]:

                continue

            tokens.append(token)

        text = "".join(tokens)

        text = text.replace(
            "▁",
            " "
        )

        return text


text = """
The transformer is a neural network architecture.
Transformers process sequences using attention.
Attention allows the model to understand relationships
between different tokens.
A language model predicts the next token.
GPT is a decoder-only transformer.
The model learns by predicting tokens from context.
Deep learning models require data and optimization.
The transformer architecture is powerful for language.
"""


words = text.lower().split()

vocab, merges = train_bpe(
    words,
    num_merges=50
)

tokenizer = BPETokenizer(
    vocab,
    merges
)


vocab_size = len(
    tokenizer.token_to_id
)
print(vocab_size)


tokens = tokenizer.encode(
    text.lower()
)

data = torch.tensor(
    tokens,
    dtype=torch.long
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

num_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"{num_params:,} parameters"
)



optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01
)


def train_step(
    model,
    optimizer,
    x,
    y
):

    model.train()

    optimizer.zero_grad()

    logits, _ = model(x)

    B, T, V = logits.shape

    loss = F.cross_entropy(
        logits.reshape(
            B * T,
            V
        ),
        y.reshape(
            B * T
        )
    )

    loss.backward()

    optimizer.step()

    return loss.item()


batch_size = 8
block_size = 64


def get_batch(
    data,
    batch_size,
    block_size,
    device
):
    """
        Why random chunks?
        Imagine our corpus:
            t0 t1 t2 t3 t4 t5 t6 t7 ...

        Instead of training only:
            t0 → t1
            t1 → t2
            t2 → t3

        we randomly select:
            t100 → t101
            t101 → t102
            ...

        then another batch:
            t350 → t351
            ...

        This gives us many different contexts
    """

    print(len(data), block_size, batch_size, len(data))

    starts = torch.randint(
        0,
        len(data) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        data[
            i:i + block_size
        ]
        for i in starts
    ])

    y = torch.stack([
        data[
            i + 1:i + block_size + 1
        ]
        for i in starts
    ])

    return (
        x.to(device),
        y.to(device)
    )


for step in range(2000):

    x, y = get_batch(
        data,
        batch_size,
        block_size,
        device
    )

    loss = train_step(
        model,
        optimizer,
        x,
        y
    )

    if step % 100 == 0:

        print(
            f"step {step}: "
            f"loss = {loss:.4f}"
        )



@torch.no_grad()
def generate(
    model,
    input_ids,
    max_new_tokens,
    temperature=1.0
):

    model.eval()

    generated = input_ids

    past_kv = None

    for _ in range(
        max_new_tokens
    ):

        if past_kv is None:

            current_input = generated

        else:

            current_input = generated[
                :, -1:
            ]

        logits, past_kv = model(
            current_input,
            past_kv
        )

        logits = logits[:, -1, :]

        logits = logits / temperature

        probs = F.softmax(
            logits,
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

    return generated



def top_k_sampling(
    logits,
    k=50
):

    values, indices = torch.topk(
        logits,
        k
    )

    filtered = torch.full_like(
        logits,
        float("-inf")
    )

    filtered.scatter_(
        -1,
        indices,
        values
    )

    return filtered





prompt = "the transformer"


prompt_ids = tokenizer.encode(
    prompt.lower()
)

input_ids = torch.tensor(
    [prompt_ids],
    dtype=torch.long,
    device=device
)

output_ids = generate(
    model,
    input_ids,
    max_new_tokens=30,
    temperature=0.8
)

output = tokenizer.decode(
    output_ids[0].tolist()
)

print(output)

model.eval()

with torch.no_grad():

    logits, _ = model(
        input_ids
    )

next_logits = logits[
    0,
    -1
]

probs = F.softmax(
    next_logits,
    dim=-1
)

top_probs, top_ids = torch.topk(
    probs,
    10
)

for prob, idx in zip(
    top_probs,
    top_ids
):

    token = tokenizer.id_to_token[
        idx.item()
    ]

    print(
        token,
        f"{prob.item():.4f}"
    )


"""
                    GPT
                     │
        ┌────────────┴────────────┐
        │                         │
     Tokenizer                 Training
        │                         │
       BPE                    Cross Entropy
        │                         │
   Token IDs                    AdamW
        │                         │
   Embeddings                     │
        │                         │
   Transformer ◄──────────────────┘
        │
   ┌────┴─────────────┐
   │                  │
  GQA                SwiGLU
   │                  │
 RoPE               FFN
   │
KV Cache
   │
   └──────────┐
              ▼
          LM Head
              │
              ▼
           Logits
              │
              ▼
         Token Sampling
              │
              ▼
            BPE
              │
              ▼
            TEXT

"""

