import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.attention = MultiHeadSelfAttention(
            d_model,
            num_heads
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

        self.ffn = FeedForward(
            d_model,
            d_ff
        )

    def forward(self, x):

        # --------------------------------
        # Attention sub-layer
        # --------------------------------

        normalized_x = self.norm1(x)

        attention_output, attention_weights = \
            self.attention(normalized_x)

        x = x + attention_output


        # --------------------------------
        # Feed-forward sub-layer
        # --------------------------------

        normalized_x = self.norm2(x)

        ffn_output = self.ffn(
            normalized_x
        )

        x = x + ffn_output


        return x, attention_weights

"""
            X
            │
            ▼
       LayerNorm
            │
            ▼
  Multi-Head Attention
            │
            ▼
            +
            ▲
            │
            X
            │
            ▼
            │
            ▼
       LayerNorm
            │
            ▼
          FFN
      16 → 64 → 16
            │
            ▼
            +
            ▲
            │
            X
            │
            ▼
         Output
"""

block = TransformerBlock(
    d_model=16,
    num_heads=4,
    d_ff=64
)

# Create some data:
x = torch.randn(
    1,
    4,
    16
)

output, weights = block(x)

print("Input:", x.shape)            # [1, 4, 16]
print("Output:", output.shape)      # [1, 4, 16]
print("Attention:", weights.shape)  # [1, 4, 4, 4]

# The Transformer block doesn't change the sequence length or model dimension.

class TransformerLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        max_seq_len
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

        self.block = TransformerBlock(
            d_model,
            num_heads,
            d_ff
        )

        self.final_norm = nn.LayerNorm(
            d_model
        )

        self.output_layer = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, x):
        # -----------------------------
        # Token embedding
        # -----------------------------

        x = self.embedding(x)

        # -----------------------------
        # Position information
        # -----------------------------

        x = self.position(x)

        # -----------------------------
        # Transformer block
        # -----------------------------

        x, attention_weights = self.block(x)

        # -----------------------------
        # Final normalization
        # -----------------------------

        x = self.final_norm(x)

        # -----------------------------
        # Vocabulary prediction
        # -----------------------------

        logits = self.output_layer(x)

        return logits, attention_weights


"""
                  Token IDs
                     │
                     ▼
                 Embedding
                     │
                     +
                     ▲
                     │
           Positional Encoding
                     │
                     ▼
              Transformer Block
                     │
          ┌──────────┴──────────┐
          │                     │
       LayerNorm              Residual
          │                     │
          ▼                     │
    Multi-Head Attention        │
          │                     │
          └──────────+──────────┘
                     │
                     ▼
                  LayerNorm
                     │
                     ▼
                    FFN
                 16 → 64 → 16
                     │
                     │
                     +
                     ▲
                     │
                   Input
                     │
                     ▼
                Final LayerNorm
                     │
                     ▼
                  Linear
                     │
                     ▼
             Vocabulary logits
                     │
                     ▼
              Next-word prediction

This is a real Transformer architecture, just a tiny one.   (Decoder Only)
"""

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

embedding_dim = 8

embedding = nn.Embedding(
    vocab_size,
    embedding_dim
)

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
    d_ff=64,
    max_seq_len=10
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
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

def predict_next_word(text):

    model.eval()

    words = text.lower().split()

    ids = [
        word_to_id[word]
        for word in words
    ]

    x = torch.tensor(
        [ids],
        dtype=torch.long
    )

    with torch.no_grad():

        logits, _ = model(x)

        last_logits = logits[:, -1, :]

        predicted_id = torch.argmax(
            last_logits,
            dim=-1
        ).item()

    return id_to_word[predicted_id]


tests = [
    "i love",
    "i love machine",
    "i love deep",
    "machine learning",
    "deep learning",
    "i study machine"
]

for text in tests:

    print(
        text,
        "→",
        predict_next_word(text)
    )

"""
The original Transformer from Attention Is All You Need has:
    Encoder
       ↓
    Decoder

But GPT-style models are decoder-only.
Our model is decoder-only because we're doing:
    previous tokens
          ↓
    causal attention
          ↓
    next token

So our architecture is conceptually:
    GPT-style Transformer
            │
            ├── Token Embedding
            │
            ├── Positional Encoding
            │
            ├── Transformer Block
            │       ├── Causal MHA
            │       ├── FFN
            │       ├── Residual
            │       └── LayerNorm
            │
            ├── ... more blocks ...
            │
            ├── Final LayerNorm
            │
            └── Language Model Head


Why do we need multiple Transformer blocks?
Real models stack many blocks:
    Embedding
       ↓
    Block 1
       ↓
    Block 2
       ↓
    Block 3
       ↓
    ...
       ↓
    Block N
       ↓
    Output
"""


