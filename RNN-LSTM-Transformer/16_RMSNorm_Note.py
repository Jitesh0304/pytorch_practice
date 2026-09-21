"""
Modern LLM architectures often use RMSNorm instead.

The difference is subtle mathematically but important architecturally.

We'll derive:
    RMS(x) = sqrt( (1 / d) * sum_i(x_i^2) + epsilon )

and then:
    RMSNorm(x) = (x / RMS(x)) * g

Then we'll replace LayerNorm -> RMSNorm in our Transformer block and see why
modern LLMs often prefer it.

Why do we normalize?
    Inside our Transformer we have activations like:
        x
         ↓
        Attention
         ↓
        Residual
         ↓
        FFN
         ↓
        Residual

    As the network gets deeper, the magnitude of activations can become difficult to control.
    Normalization keeps the activations in a more stable range.

#-------------------------------------------
#       LayerNorm
#-------------------------------------------
LayerNorm calculates both:
    * mean
    * variance

For a vector:
    x = [x_1, x_2, ..., x_d]

the mean is:
    mu = (1 / d) * sum_i(x_i)

and variance:
    sigma^2 = (1 / d) * sum_i((x_i - mu)^2)

Then:
    LayerNorm(x) = ((x - mu) / sqrt(sigma^2 + epsilon)) * gamma + beta

So LayerNorm asks:
    What is the mean and variance of this vector?

#-------------------------------------------
#       RMSNorm
#-------------------------------------------

RMSNorm only looks at the magnitude.
    First calculate:
        RMS(x) = sqrt( (1 / d) * sum_i(x_i^2) + epsilon )

    Then normalize:
        RMSNorm(x) = (x / RMS(x)) * g

Notice what disappeared:

    LayerNorm:
        mean
         +
        variance
         ↓
        normalize

    RMSNorm:
        squared values
         ↓
        mean square
         ↓
        square root
         ↓
        normalize

    There is no mean subtraction.

Suppose:
    x = [2, 4, 6, 8]

Calculate:
    RMS(x) = sqrt( (2^2 + 4^2 + 6^2 + 8^2) / 4 )
           = sqrt( (4 + 16 + 36 + 64) / 4 )
           = sqrt(30)

Approximately:
    5.477

So:
    x / RMS

becomes approximately:
    [0.365, 0.730, 1.095, 1.461]

Then RMSNorm has a learnable scale parameter.
"""


import torch
import torch.nn as nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model,
        eps=1e-8
    ):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(d_model)
        )

    def forward(self, x):
        # x.shape = [batch, sequence, d_model]

        rms = torch.sqrt(
            x.pow(2).mean(
                dim=-1,
                keepdim=True
            )
            + self.eps
        )

        x = x / rms

        return (
            x * self.weight
        )


norm = RMSNorm(4)

x = torch.tensor([
    [2.0, 4.0, 6.0, 8.0]
])

y = norm(x)

print(y)


"""
Why is weight learnable?

We don't want to force every dimension to have exactly the same scale forever.
So RMSNorm has:
    g = [g_1, g_2, ..., g_d]
and:
    y_i = g_i * (x_i / RMS(x))

The model learns these values during training.
Initially:
    g = [1, 1, 1, ...]
but during training they change.


Very small difference from the outside.

But mathematically:

    LayerNorm
        │
        ├── mean
        ├── variance
        └── normalize
    
    RMSNorm
        │
        └── RMS
             │
             └── normalize
"""

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

        self.ffn = FeedForward(
            d_model,
            d_ff
        )

def forward(
    self,
    x,
    padding_mask=None
):

    normalized_x = self.norm1(x)

    attention_output, weights, K, V = \
        self.attention(
            normalized_x,
            padding_mask
        )

    x = x + attention_output

    normalized_x = self.norm2(x)

    ffn_output = self.ffn(
        normalized_x
    )

    x = x + ffn_output

    return x


"""
We're doing:
    x
     │
     ▼
    RMSNorm
     │
     ▼
    Attention
     │
     ▼
    +
     ▲
     │
     x


Our block is:
        x
        │
        ├───────────────┐
        │               │
        ▼               │
     RMSNorm            │
        │               │
        ▼               │
    Attention           │
        │               │
        ▼               │
        +◄──────────────┘
        │
        ├───────────────┐
        │               │
        ▼               │
     RMSNorm            │
        │               │
        ▼               │
       FFN              │
        │               │
        ▼               │
        +◄──────────────┘
        │
        ▼
       output


Why Pre-Norm?
    One major reason is training stability, especially as Transformers become deeper.

Think about the residual stream:
    x_l + 1 = x_l + F(Norm(x_l))

The original information can flow through the residual connection relatively directly.

This gives us:
    x
    │
    ├──────────────►
    │               │
    ▼               ▼
    Norm → Block → +

The residual path acts like a highway through the network.
This becomes particularly valuable when we stack dozens or hundreds of layers.

"""