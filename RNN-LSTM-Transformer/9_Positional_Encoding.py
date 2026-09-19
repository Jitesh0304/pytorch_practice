"""
"i love machine learning"

why the Transformer needs to know that:
    i = position 0
    love = position 1
    machine = position 2
    learning = position 3

The problem: attention doesn't know position.

Consider:
    I love machine learning

versus:
    machine learning love I

They contain the same words but have completely different meanings.
We therefore need to give every token position information.

Instead of
    X = Token Embedding
We use
    X = Token Embedding + Positional Encoding

So:
    Token
      ↓
    Embedding ───────┐
                     +
    Position ────────┘
           ↓
    Transformer

For example:
    "machine"

token embedding:
[0.21, 0.73, 0.12, ...]

position encoding:
[0.91, 0.14, 0.84, ...]
              ↓ +
final representation:
[1.12, 0.87, 0.96, ...]

The model now receives both:
    what the token is
    where the token is

PE(pos, 2i)   = sin(pos / (10000 ** (2i / d_model)))
PE(pos, 2i+1) = cos(pos / (10000 ** (2i / d_model)))

It simply means:
    even dimensions → sin
    odd dimensions  → cos

with different frequencies.

Suppose:
    d_model = 4

and our sequence is:
    i love machine learning

Positions are:
    i        → 0
    love     → 1
    machine  → 2
    learning → 3

Our positional encoding has shape:
    [4 positions, 4 dimensions]

Conceptually:
                 dimension
              0      1      2      3
           ┌──────┬──────┬──────┬──────┐
    pos 0  │ sin  │ cos  │ sin  │ cos  │
    pos 1  │ sin  │ cos  │ sin  │ cos  │
    pos 2  │ sin  │ cos  │ sin  │ cos  │
    pos 3  │ sin  │ cos  │ sin  │ cos  │
           └──────┴──────┴──────┴──────┘

"""

import torch
import math

def positional_encoding(
    seq_len,
    d_model
):

    PE = torch.zeros(
        seq_len,
        d_model
    )

    for pos in range(seq_len):

        for i in range(0, d_model, 2):

            PE[pos, i] = math.sin(
                pos /
                (10000 ** (i / d_model))
            )

            PE[pos, i + 1] = math.cos(
                pos /
                (10000 ** (i / d_model))
            )

    return PE


PE = positional_encoding(
    seq_len=4,
    d_model=8
)

print(PE)

"""
Look at position 0.

Since:
    sin(0) = 0

and:
    cos(0) = 1

we get:
position 0:
    [0, 1, 0, 1, 0, 1, 0, 1]

That's perfectly normal.
Position 1 gets a different pattern.
Position 2 gets another pattern.

And so on.
Therefore each position has a unique mathematical signal.


This is a clever part.
    We don't simply use:
        position = 0
        position = 1
        position = 2
        position = 3

because a single scalar position isn't enough to create a rich representation.
Instead, we create multiple waves with different frequencies.

Think of it like this:
    Dimension 0:
    sin ────~~~~──~~~~──~~~~
    
    Dimension 2:
    sin ──~~──~~──~~──~~──
    
    Dimension 4:
    sin ─~~~~──~~~~──~~~~
    
    Dimension 6:
    sin ─~─~─~─~─~─~─~─~─

Different dimensions change at different speeds.
So each position gets a distinctive combination.

Imagine each dimension is a clock.
At position 0:
    Clock 1 → 0°
    Clock 2 → 0°
    Clock 3 → 0°
    Clock 4 → 0°

At position 1:
    Clock 1 → small movement
    Clock 2 → different movement
    Clock 3 → slower movement
    Clock 4 → even slower movement

At position 100:
all clocks have different phases
The combination of these phases tells the model approximately where the token is.

Suppose 
    X.shape = [4, 8]
    PE.shape = [4, 8]

Therefore we can simply do:
    X_with_position = X + PE

Now:
    X
     ↓
    Token information
    
    PE
     ↓
    Position information
    
    X + PE
     ↓
    Token + Position
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_seq_len=512
    ):
        super().__init__()

        PE = torch.zeros(
            max_seq_len,
            d_model
        )

        position = torch.arange(
            0,
            max_seq_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            * (
                -math.log(10000.0)
                / d_model
            )
        )

        PE[:, 0::2] = torch.sin(
            position * div_term
        )

        PE[:, 1::2] = torch.cos(
            position * div_term
        )

        PE = PE.unsqueeze(0)

        self.register_buffer(
            "PE",
            PE
        )

    def forward(self, x):

        seq_len = x.size(1)

        return x + self.PE[:, :seq_len]

"""
This line:
    self.register_buffer(
        "PE",
        PE
    )

is important.

The positional encoding isn't a parameter that the optimizer should learn.
We don't want:
    PE → gradient → optimizer → change PE

Instead:
    PE → fixed
    register_buffer tells PyTorch:
        Keep this tensor as part of the model, move it with the model to CPU/GPU, but don't train it as a parameter.

"""

d_model = 16

position = PositionalEncoding(
    d_model=d_model,
    max_seq_len=100
)

torch.manual_seed(42)

sentences = [
    "i love machine learning",
    "i love deep learning",
    "machine learning is powerful",
    "deep learning is interesting",
    "i study machine learning"
]

# Vocabulary

tokenized = [
    sentence.split()
    for sentence in sentences
]

vocab = sorted(
    set(
        word
        for sentence in tokenized
        for word in sentence
    )
)

word_to_id = {
    word: i
    for i, word in enumerate(vocab)
}

id_to_word = {
    i: word
    for word, i in word_to_id.items()
}

vocab_size = len(vocab)

embedding = nn.Embedding(
    vocab_size,
    d_model
)

sentence = "i love machine learning"

tokens = sentence.split()

ids = torch.tensor([
    word_to_id[word]
    for word in tokens
])

X = embedding(ids)

# Add batch dimension
X = X.unsqueeze(0)

# print(X.shape)        # [1, 4, 16]

X_positioned = position(X)

# print(X_positioned.shape)     # [1, 4, 16]

"""
Previously we had:
    Embedding
        ↓
    Multi-Head Attention
        ↓
    Linear

Now:
    Embedding
        ↓
    Positional Encoding
        ↓
    Multi-Head Attention
        ↓
    Linear
"""


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads

        self.head_dim = (
            d_model // num_heads
        )

        # Q, K, V projections

        self.W_Q = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.W_K = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.W_V = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        # Final projection

        self.W_O = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

    def forward(self, X):

        batch_size, seq_len, _ = X.shape

        # --------------------------------
        # Q, K, V
        # --------------------------------

        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)


        # --------------------------------
        # Split into heads
        # --------------------------------

        Q = Q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        K = K.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        V = V.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )


        # --------------------------------
        # Move heads before sequence
        # --------------------------------

        Q = Q.transpose(1, 2)

        K = K.transpose(1, 2)

        V = V.transpose(1, 2)


        # --------------------------------
        # Attention scores
        # --------------------------------

        scores = Q @ K.transpose(-2, -1)

        # --------------------------------
        # Scale
        # --------------------------------

        scores = scores / (
            self.head_dim ** 0.5
        )


        # --------------------------------
        # Causal mask
        # --------------------------------

        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=X.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # --------------------------------
        # Softmax
        # --------------------------------

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # --------------------------------
        # Weighted sum
        # --------------------------------

        output = attention_weights @ V

        # --------------------------------
        # Combine heads
        # --------------------------------

        output = output.transpose(
            1,
            2
        )

        output = output.contiguous().view(
            batch_size,
            seq_len,
            self.d_model
        )

        # --------------------------------
        # Final projection
        # --------------------------------

        output = self.W_O(output)

        return output, attention_weights


class TransformerLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        max_seq_len=512
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.position = PositionalEncoding(
            d_model,
            max_seq_len
        )

        self.attention = MultiHeadSelfAttention(
            d_model,
            num_heads
        )

        self.output_layer = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, x):

        # Token embeddings
        x = self.embedding(x)

        # Add position
        x = self.position(x)

        # Multi-head causal attention
        x, attention_weights = self.attention(x)

        # Vocabulary prediction
        logits = self.output_layer(x)

        return logits, attention_weights


"""
Now we have:

Token IDs
    ↓
Embedding
    ↓
+ Positional Encoding
    ↓
Multi-Head Causal Self-Attention
    ↓
Linear
    ↓
Vocabulary probabilities
"""

X_data = []
Y_data = []

for sentence in tokenized:

    ids = [
        word_to_id[word]
        for word in sentence
    ]

    X_data.append(ids[:-1])
    Y_data.append(ids[1:])

X_data = torch.tensor(X_data)
Y_data = torch.tensor(Y_data)

model = TransformerLanguageModel(
    vocab_size=vocab_size,
    d_model=16,
    num_heads=4,
    max_seq_len=10
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

for epoch in range(1000):

    logits, attention_weights = model(
        X_data
    )

    loss = criterion(
        logits.reshape(-1, vocab_size),
        Y_data.reshape(-1)
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 100 == 0:

        print(
            f"Epoch {epoch + 1:4d} "
            f"Loss: {loss.item():.4f}"
        )

"""
We have now built:

                Input
                  │
                  ▼
             Token IDs
                  │
                  ▼
              Embedding
                  │
                  │
                  ├───────────────┐
                  │               │
                  ▼               ▼
             Token vector   Position vector
                  │               │
                  └───────┬───────┘
                          │
                          ▼
                    Add together
                          │
                          ▼
                 Multi-Head Attention
                          │
                          ▼
                       W_O
                          │
                          ▼
                       Linear
                          │
                          ▼
                   Vocabulary logits
                          │
                          ▼
                       Softmax
                          │
                          ▼
                    Next word
"""


