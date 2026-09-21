import math
import torch
import torch.nn as nn
import torch.nn.functional as F


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(device)


# vocab_size     = 5000
# d_model        = 128
# num_layers     = 4
# num_q_heads    = 8
# num_kv_heads   = 2
# ffn_dim        = 256
# max_seq_len    = 128



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


model = GPT(
    vocab_size=5000,
    d_model=128,
    num_layers=4,
    num_q_heads=8,
    num_kv_heads=2,
    ffn_dim=256,
    max_seq_len=128
).to(device)

num_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"{num_params:,} parameters"
)

input_ids = torch.tensor(
    [[
        2,
        15,
        42,
        71,
        93
    ]],
    device=device
)

logits, kv_cache = model(
    input_ids
)

next_token_logits = logits[
    :, -1, :
]

probs = F.softmax(
    next_token_logits,
    dim=-1
)

next_token = torch.argmax(
    probs,
    dim=-1
)

print(next_token)

"""
Suppose the encoded sequence is:
    [10, 25, 42, 71, 93]

GPT training works like:
    Input:
        10 25 42 71
    Target:
        25 42 71 93

In other words:
    current token
          ↓
    predict next token

This is called next-token prediction.
"""


input_ids = torch.tensor(
    [[10, 25, 42, 71]],
    device=device
)

targets = torch.tensor(
    [[25, 42, 71, 93]],
    device=device
)

logits, _ = model(
    input_ids
)

loss = F.cross_entropy(
    logits.view(-1, model.vocab_size),
    targets.view(-1)
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)

for step in range(1000):

    optimizer.zero_grad()

    logits, _ = model(
        input_ids
    )

    loss = F.cross_entropy(
        logits.view(
            -1,
            model.vocab_size
        ),
        targets.view(-1)
    )

    loss.backward()

    optimizer.step()

    if step % 100 == 0:

        print(
            step,
            loss.item()
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


logits = top_k_sampling(
    logits,
    k=50
)

probs = F.softmax(
    logits / temperature,
    dim=-1
)

"""
Our final pipeline is:

                 RAW TEXT
                    │
                    ▼
                BPE TOKENIZER
                    │
                    ▼
               [token IDs]
                    │
                    ▼
             TOKEN EMBEDDING
                    │
                    ▼
          ┌─────────────────────┐
          │ Transformer Block 1  │
          │                     │
          │ RMSNorm             │
          │ GQA + RoPE          │
          │ Residual            │
          │ RMSNorm             │
          │ SwiGLU              │
          │ Residual            │
          └──────────┬──────────┘
                     │
                    ...
                     │
          ┌──────────▼──────────┐
          │ Transformer Block N │
          └──────────┬──────────┘
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
                Softmax/Sampling
                     │
                     ▼
                Next Token
                     │
                     ▼
               KV Cache update
                     │
                     └───────► repeat
"""


