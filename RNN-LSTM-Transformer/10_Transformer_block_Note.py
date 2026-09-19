"""
The architecture we'll build is:

                  Input X
                     │
                     ▼
              Layer Normalization
                     │
                     ▼
          Multi-Head Self-Attention
                     │
                     ▼
                  + X ◄──────────────┐
                     │               │
                     ▼               │
              Layer Normalization    │
                     │               │
                     ▼               │
            Feed-Forward Network     │
                     │               │
                     ▼               │
                  + X ───────────────┘
                     │
                     ▼
                   Output

Mathematically -:
    A = X + MHA(LN(X))          MHA = Multihead-Attentions, LN = Layer Normalization
then
    Output = A + FFN(LN(A))     FFN = Feed-Forward Network

FFN(X) = W_2 GELU(W1x + b1) + b2
For our model:
    d_model = 16

we might expand to:
    16 → 64 → 16

Why expand?
    Because the network gets a larger internal representation in which it can perform more complex transformations.


"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            d_model,
            d_ff
        )

        self.fc2 = nn.Linear(
            d_ff,
            d_model
        )

    def forward(self, x):

        x = self.fc1(x)

        x = F.gelu(x)

        x = self.fc2(x)

        return x

ffn = FeedForward(
    d_model=16,
    d_ff=64
)
"""
The flow is:

[batch, sequence, 16]
          ↓
       Linear
          ↓
[batch, sequence, 64]
          ↓
        GELU
          ↓
       Linear
          ↓
[batch, sequence, 16]


Previously, with simpler neural networks, we often used:
    F.relu(x)

Transformers commonly use:
    F.gelu(x)

GELU is smoother than ReLU.
Conceptually:
    ReLU:
        negative → 0
        positive → unchanged

    GELU behaves more smoothly:
        small negative → mostly suppressed
        small positive → partially retained
        large positive → mostly retained
        

The exact function is:
    GELU(x) = xΦ(x)

    where Φ(x)
         is the standard normal cumulative distribution function.

For now, remember:
    GELU = nonlinear activation commonly used in Transformers

We also need:
    nn.LayerNorm(d_model)

Why?
    Our representations can have different scales during training.
    LayerNorm normalizes the features of each token representation.

For a vector:
    [2.1, 5.4, -1.2, 0.7]

LayerNorm roughly transforms it into a normalized representation with controlled mean and variance.

Why residual connections?
    This is another extremely important Transformer concept.
    Instead of:
        X → Attention → Output
    we use:
        X ───────────────┐
                         +
        Attention(X) ────┘
                         ↓
                       Output
    So:
        Output = X + Attention(X)
    
    This is called a residual connection or skip connection.

Imagine the attention layer learns a transformation:
    Attention(X)

    Instead of forcing the layer to learn an entirely new representation, it only needs to learn a useful change 
    to the existing representation.

The model can preserve information:
    original information
           +
    new information

This makes deep networks significantly easier to optimize.
"""









