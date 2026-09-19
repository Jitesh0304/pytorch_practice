"""
You can now understand this diagram:

                    Transformer
                        │
          ┌─────────────┴─────────────┐
          │                           │
       Embedding                 Position
          │                           │
          └─────────────┬─────────────┘
                        ↓
                Transformer Block
                        │
             ┌──────────┴──────────┐
             │                     │
        Causal MHA                FFN
             │                     │
             └──────────┬──────────┘
                        ↓
                  Residual + Norm
                        ↓
                  Next Transformer
                        ↓
                       ...
                        ↓
                  Language Head
                        ↓
                   Next Token

Embedding
    ↓
Positional Encoding
    ↓
Transformer Block 1
    ↓
Transformer Block 2
    ↓
Transformer Block 3
    ↓
...
    ↓
Transformer Block N
    ↓
Final LayerNorm
    ↓
Linear
    ↓
Next-token prediction

The important point is:
    Every block has the same basic architecture, but each block has its own learned parameters.

a simplified intuition is:
    Embedding
       ↓
    basic token information

    Block 1
       ↓
    local relationships

    Block 2
       ↓
    higher-level relationships

    Block 3
       ↓
    more abstract representation

    Block N
       ↓
    representation useful for prediction

This isn't a strict rule that "Block 1 learns syntax and Block 2 learns semantics"—real networks are more distributed
than that—but stacking lets the model repeatedly transform and mix information.
"""

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


class GPTModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        max_seq_len
    ):
        super().__init__()

        # Token embeddings

        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        # Positional encoding

        self.position = PositionalEncoding(
            d_model,
            max_seq_len
        )

        # Transformer blocks

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff
            )
            for _ in range(num_layers)
        ])

        # Final normalization

        self.final_norm = nn.LayerNorm(
            d_model
        )

        # Language model head

        self.output_layer = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, x):

        # --------------------------
        # Token embedding
        # --------------------------

        x = self.embedding(x)

        # --------------------------
        # Position
        # --------------------------

        x = self.position(x)

        # --------------------------
        # Transformer blocks
        # --------------------------

        attention_weights = []

        for block in self.blocks:

            x, weights = block(x)

            attention_weights.append(
                weights
            )

        # --------------------------
        # Final normalization
        # --------------------------

        x = self.final_norm(x)

        # --------------------------
        # Vocabulary logits
        # --------------------------

        logits = self.output_layer(x)

        return logits, attention_weights


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

model = GPTModel(
    vocab_size=vocab_size,
    d_model=32,
    num_heads=4,
    d_ff=128,
    num_layers=4,
    max_seq_len=20
)

"""
                  Token IDs
                     │
                     ▼
                 Embedding
                  32 dim
                     │
                     ▼
            Positional Encoding
                     │
                     ▼
             ┌───────────────┐
             │ Transformer   │
             │ Block 1       │
             └───────────────┘
                     │
                     ▼
             ┌───────────────┐
             │ Transformer   │
             │ Block 2       │
             └───────────────┘
                     │
                     ▼
             ┌───────────────┐
             │ Transformer   │
             │ Block 3       │
             └───────────────┘
                     │
                     ▼
             ┌───────────────┐
             │ Transformer   │
             │ Block 4       │
             └───────────────┘
                     │
                     ▼
                LayerNorm
                     │
                     ▼
                  Linear
                     │
                     ▼
                Vocabulary
"""

# sentence = "i love machine"
#
# ids = torch.tensor([
#     [
#         word_to_id["i"],
#         word_to_id["love"],
#         word_to_id["machine"]
#     ]
# ])
#
# logits, attention_weights = model(ids)
#
# print(logits.shape)     # [1, 3, vocab_size]

"""
Why?
    Because the model produces a prediction for every position.

For:
    i love machine

we get:
    position 0 → prediction
    position 1 → prediction
    position 2 → prediction

What are those predictions?
    i
    ↓
    love
    
    i love
    ↓
    machine
    
    i love machine
    ↓
    learning

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


"""
Conceptually:
    X:                                       Y:
    
    i love machine                          love machine learning
    i love deep                             love deep learning
    machine learning is                     learning is powerful
    deep learning is                        learning is interesting
    i study machine                         study machine learning

This is next-token prediction.
"""

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.003
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


def generate(
    model,
    text,
    max_new_tokens
):

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

    for _ in range(max_new_tokens):

        with torch.no_grad():

            logits, _ = model(x)

        # Last token's prediction

        next_logits = logits[:, -1, :]

        # Highest probability token

        next_id = torch.argmax(
            next_logits,
            dim=-1
        ).item()

        # Append token

        next_token = torch.tensor(
            [[next_id]]
        )

        x = torch.cat(
            [x, next_token],
            dim=1
        )

    return [
        id_to_word[token_id]
        for token_id in x[0].tolist()
    ]


result = generate(
    model,
    "i love",
    max_new_tokens=2
)

print(result)


"""
Suppose we start:
    i love

The model sees:
    i love

and predicts:
    machine

Now we feed:
    i love machine

The model predicts:
    learning

Then:
    i love machine learning

This is autoregressive generation.
This equation is fundamental to GPT-style models.
The model estimates:
    P(next token | previous tokens)
again and again.

But argmax isn't how modern generation usually works
Our current generation:
    next_id = torch.argmax(
        next_logits,
        dim=-1
    )

always chooses the single most probable token. That's called greedy decoding.

It's deterministic.
    But language generation benefits from sampling.

Suppose the model predicts:
    machine     0.60
    deep        0.30
    data        0.07
    python      0.03

Greedy:
    machine

Sampling:
    machine → 60%
    deep    → 30%
    data    → 7%
    python  → 3%

Sampling can generate different continuations.

Softmax
    Our model outputs logits, not probabilities.

For example:
    machine → 4.2
    deep    → 3.5
    data    → 1.8
    python  → 0.4


P_i = exp(z_i) / sum(exp(z_j))
    Where:
      P_i  = Probability of class i
      z_i  = Logit score for class i
      sum  = Summation over all classes j

using:
    probs = F.softmax(
        logits,
        dim=-1
    )

Now:
    machine → 0.60
    deep    → 0.30
    data    → 0.07
    python  → 0.03
    
Random sampling
    PyTorch provides:
        torch.multinomial()

    So:
        next_id = torch.multinomial(
            probs,
            num_samples=1
        ).item()

Now we're sampling according to the probability distribution.

Add temperature (T) to control randomness.

We modify logits:
    z' = z / T

Then apply Softmax:
    P = softmax(z')

Behavior:
  If T < 1: The distribution becomes sharper (more confident/deterministic).
  If T > 1: The distribution becomes flatter (more random/diverse).

Low temperature
    T = 0.2

More deterministic:
    machine  0.95
    deep     0.04
    data     0.01

Higher temperature
    T = 1.5

More random:
    machine  0.45
    deep     0.30
    data     0.15
    python   0.10

"""


def generate(
    model,
    text,
    max_new_tokens,
    temperature=1.0
):

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

    for _ in range(max_new_tokens):

        with torch.no_grad():

            logits, _ = model(x)

        # Last position

        next_logits = logits[:, -1, :]

        # Temperature

        next_logits = (
            next_logits / temperature
        )

        # Convert to probabilities

        probs = F.softmax(
            next_logits,
            dim=-1
        )

        # Sample

        next_id = torch.multinomial(
            probs,
            num_samples=1
        )

        # Append

        x = torch.cat(
            [x, next_id],
            dim=1
        )

    return [
        id_to_word[token_id]
        for token_id in x[0].tolist()
    ]


torch.manual_seed(42)

print(
    generate(
        model,
        "i love",
        2,
        temperature=0.5
    )
)


"""
One more important concept: weight tying
Our model currently has:

Embedding:
    vocab_size → d_model

and:
    Output layer:
        d_model → vocab_size

So we have two sets of vocabulary-related parameters.
Many language models tie these weights.

Conceptually:
    Embedding
        ↑
        │
    same weights
        │
        ↓
    Output projection

Instead of:
    self.output_layer = nn.Linear(
        d_model,
        vocab_size
    )

we can use the embedding matrix.

We have now built:

                    Input tokens
                         │
                         ▼
                    Embedding
                         │
                         ▼
              Positional Encoding
                         │
                         ▼
                ┌─────────────────┐
                │ Transformer     │
                │ Block 1         │
                │                 │
                │ LN              │
                │ ↓               │
                │ Causal MHA      │
                │ ↓               │
                │ Residual        │
                │ ↓               │
                │ LN              │
                │ ↓               │
                │ FFN             │
                │ ↓               │
                │ Residual        │
                └─────────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Transformer     │
                │ Block 2         │
                └─────────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Transformer     │
                │ Block 3         │
                └─────────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Transformer     │
                │ Block 4         │
                └─────────────────┘
                         │
                         ▼
                    Final LN
                         │
                         ▼
                  Output Linear
                         │
                         ▼
                     Logits
                         │
                         ▼
                      Softmax
                         │
                         ▼
                   Next token
                         │
                         └──────────┐
                                    │
                                    ▼
                              Feed back in

That feedback loop is what gives us autoregressive text generation.
"""

