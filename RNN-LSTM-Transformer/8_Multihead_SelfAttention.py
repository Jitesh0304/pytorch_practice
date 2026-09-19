"""
Instead of one attention mechanism:
             Attention
                 ↓
              output

we'll have:
       ┌───────────────┐
       │               │
    Head 1          Head 2
       │               │
    Head 3          Head 4
       │               │
       └───────┬───────┘
               ↓
           Concatenate
               ↓
             Linear
Different heads can learn different relationships.
    Q, K, V ---> multiple head ---> concatenate ---> W_0

SelfAttention has
    W_Q, W_K, W_V

So we have one attention mechanism.
But language contains many kinds of relationships.
For example:
    "I love machine learning"

One attention head might learn relationships like:
    i → love

Another might learn:
    machine → learning

Another might learn positional or syntactic relationships.
    So instead of:
                 X
                 ↓
          Self-Attention
                 ↓
              Output

Multihead Attention
                    X
                    │
        ┌───────────┼───────────┐
        ↓           ↓           ↓
      Head 1      Head 2      Head 3
        ↓           ↓           ↓
      Attn 1      Attn 2      Attn 3
        │           │           │
        └───────────┼───────────┘
                    ↓
              Concatenate
                    ↓
                  W_O
                    ↓
                 Output

Suppose our model dimension is:
    d_model = 8

and we use:
    h = 2

attention heads.
    We split the representation:
    8 ÷ 2 = 4

So each head works with:
    d_k = 4

dimensions.
    X
    [sequence, 8]
          │
          ├──── Head 1 → [sequence, 4]
          │
          └──── Head 2 → [sequence, 4]

Then concatenate:
    [sequence, 4] + [sequence, 4]

to get
    [sequence, 8]

Multihead (X) = Concat(head_1, ..., head_n)W_0

"""


import torch
import torch.nn as nn
import torch.nn.functional as F

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

sentence = "i love machine learning"

tokens = sentence.split()

ids = torch.tensor([
    word_to_id[word]
    for word in tokens
])

X = embedding(ids)

# print("X shape:", X.shape)    # [4, 8] --> 4 tokens, 8 dimensional representation

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



attention = MultiHeadSelfAttention(
    d_model=8,
    num_heads=2
)

X_batch = X.unsqueeze(0)
# print(X_batch.shape)        # [1, 4, 8] --> batch = 1, sequence = 4, embedding = 8

output, weights = attention(
    X_batch
)

print("Output:", output.shape)      # [1, 4, 8]
print("Weights:", weights.shape)        # [1, 2, 4, 4]  ---> [batch, heads, query, key]
"""
1 = batch
2 = attention heads
4 = query tokens
4 = key tokens
Therefore each head has its own:
    4 × 4
attention matrix.
"""

head_1 = weights[0, 0]
# print(head_1)       # [4, 4]

# head_2 = weights[0, 1]

words = tokens

print(
    "             "
    + " ".join(
        f"{word:>10}"
        for word in words
    )
)

for i, word in enumerate(words):

    print(
        f"{word:>10} "
        + " ".join(
            f"{value.item():10.3f}"
            for value in head_1[i]
        )
    )


"""
Head_1 and Head_2 has completely different attention distribution.

That's the whole point.
    Head 1
       ↓
    learns one set of relationships
    
    Head 2
       ↓
    learns another set of relationships


Input :
    X = [4, 8]

After Q projection :
    Q = [4, 8]

But with 2 heads:

    Q = [4, 8]
       ↓ split

    Head 1
        [4, 4]
    
    Head 2
        [4, 4]

Same for K and V.
Each head independently calculates:

So:
             X
             │
        ┌────┴────┐
        ↓         ↓
      Head 1    Head 2
        ↓         ↓
      Q₁K₁ᵀ     Q₂K₂ᵀ
        ↓         ↓
     Softmax   Softmax
        ↓         ↓
      V₁         V₂
        ↓         ↓
      out₁      out₂
        └────┬────┘
             ↓
       Concatenate
             ↓
           [4,8]
             ↓
            W_O
             ↓
          [4,8]

After concatenating:

head 1 → 4 dimensions
head 2 → 4 dimensions

concatenate → 8 dimensions

But we want the model to be able to mix information across heads.

That's what:
    W_O

does.
    Head 1 ──┐
             ├──→ Concatenate → W_O → Output
    Head 2 ──┘

So:
    Multihead (X) = Concat(head_1, head_2)W_0
"""

class TransformerLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model
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

        x = self.embedding(x)

        x, attention_weights = self.attention(x)

        logits = self.output_layer(x)

        return logits, attention_weights

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

# print(X_data.shape)     # [5, 3]
# print(Y_data.shape)     # [5, 3]


model = TransformerLanguageModel(
    vocab_size=vocab_size,
    d_model=16,         # embedding size
    num_heads=4
)

"""
Notice:
    16 / 4 = 4
So every head gets:
    head_dim = 4
    
"""

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

# Now inspect the trained heads
text = "i love machine learning"

ids = [
    word_to_id[word]
    for word in text.split()
]

x = torch.tensor(
    [ids],
    dtype=torch.long
)

model.eval()

with torch.no_grad():

    logits, attention_weights = model(x)

print(attention_weights.shape)          # [1, 4, 4, 4]
"""
batch = 1
heads = 4
query = 4
key = 4

We now have four different attention matrices.
"""

for head in range(4):

    print(
        f"\n========== HEAD {head + 1} =========="
    )

    weights = attention_weights[
        0,
        head
    ]

    print(
        "             "
        + " ".join(
            f"{word:>10}"
            for word in text.split()
        )
    )

    for i, word in enumerate(
        text.split()
    ):

        print(
            f"{word:>10} "
            + " ".join(
                f"{value.item():10.3f}"
                for value in weights[i]
            )
        )

# After sufficient training on a meaningful dataset, different heads can specialize in different patterns.

