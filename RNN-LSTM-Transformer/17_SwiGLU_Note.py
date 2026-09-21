"""
Mathematically:
    FFN(x) = W_2 * GELU(W_1 * x)

SwiGLU changes this into a gated feed-forward network:
    SwiGLU(x) = W_o * [ SiLU(x * W_g) * (x * W_v) ]

where:
    * W_g = gate projection
    * W_v = value projection
    * W_o = output projection
    * *   = element-wise multiplication

The key idea is:
    One branch decides what information should pass through, while another branch provides the information.


First understand SiLU
SwiGLU uses SiLU.

SiLU is:
    SiLU(x) = x * sigma(x)

where:
    sigma(x) = 1 / (1 + e^(-x))

So:
    SiLU(x) = x * (1 / (1 + e^(-x)))

Example:
  x       sigmoid(x)      SiLU(x)
    -2       0.119          -0.238
    -1       0.269          -0.269
     0       0.500           0
     1       0.731           0.731
     2       0.881           1.762
Unlike ReLU, SiLU is smooth.


Why do we need a gate?
    Consider a normal FFN:
        h = GELU(x * W_1)

    Every dimension is transformed independently after the projection.

SwiGLU creates two representations:
                 ┌── Gate branch ── SiLU ──┐
                 │                          │
x ───────────────┤                          × ──►
                 │                          │
                 └── Value branch ─────────┘

Specifically:
    G = SiLU(x * W_g)

and:
    V = x * W_v

then:
    H = G * V

The gate can suppress or amplify parts of the value representation.

"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff
    ):
        super().__init__()

        self.gate_proj = nn.Linear(
            d_model,
            d_ff,
            bias=False
        )

        self.value_proj = nn.Linear(
            d_model,
            d_ff,
            bias=False
        )

        self.output_proj = nn.Linear(
            d_ff,
            d_model,
            bias=False
        )

    def forward(self, x):

        gate = self.gate_proj(x)

        value = self.value_proj(x)

        gate = F.silu(gate)

        hidden = gate * value

        output = self.output_proj(
            hidden
        )

        return output

"""
Suppose:
    batch = 10
    sequence = 12
    d_model = 64
    d_ff = 256

Input:
    [10, 12, 64]

Gate branch:
    [10, 12, 64]
           ↓
    Linear
           ↓
    [10, 12, 256]

Value branch:
    [10, 12, 64]
           ↓
    Linear
           ↓
    [10, 12, 256]

Then:
    SiLU(gate)
          ×
       value
          ↓
    [10, 12, 256]
    
Then:
    Linear
      ↓
    [10, 12, 64]

So the input/output dimensionality remains:
    64 → 64

while the internal representation expands to:
    64 → 256 → 64

"""


class SwiGLU(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff
    ):
        super().__init__()

        self.gate_proj = nn.Linear(
            d_model,
            d_ff,
            bias=False
        )

        self.value_proj = nn.Linear(
            d_model,
            d_ff,
            bias=False
        )

        self.output_proj = nn.Linear(
            d_ff,
            d_model,
            bias=False
        )

    def forward(self, x):

        gate = F.silu(
            self.gate_proj(x)
        )

        value = self.value_proj(x)

        hidden = gate * value

        return self.output_proj(
            hidden
        )



class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        max_seq_len
    ):
        super().__init__()

        self.norm1 = RMSNorm(
            d_model
        )

        self.attention = MultiHeadSelfAttention(
            d_model,
            num_heads,
            max_seq_len
        )

        self.norm2 = RMSNorm(
            d_model
        )

        self.ffn = SwiGLU(
            d_model,
            d_ff
        )

    def forward(
        self,
        x,
        padding_mask=None
    ):

        # Attention

        residual = x

        x = self.norm1(x)

        attention_output, weights, K, V = \
            self.attention(
                x,
                padding_mask
            )

        x = residual + attention_output

        # SwiGLU

        residual = x

        x = self.norm2(x)

        x = self.ffn(x)

        x = residual + x

        return x

"""
                    x
                    │
             ┌──────┴──────┐
             │             │
             │         Residual
             │             │
             ▼             │
          RMSNorm          │
             │             │
             ▼             │
          Attention         │
             │             │
             └──────► + ◄──┘
                       │
                       │
                ┌──────┴──────┐
                │             │
                │         Residual
                │             │
                ▼             │
             RMSNorm          │
                │             │
                ▼             │
              SwiGLU           │
                │             │
                └──────► + ◄──┘
                              │
                              ▼
                            output


Current model
        Token Embedding
              ↓
        RMSNorm
              ↓
        Multi-Head Attention
              │
              ├── RoPE(Q)
              ├── RoPE(K)
              └── KV Cache
              ↓
        Residual
              ↓
        RMSNorm
              ↓
        SwiGLU
              ↓
        Residual
"""


