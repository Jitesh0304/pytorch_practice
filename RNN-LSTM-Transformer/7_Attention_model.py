import torch
import torch.nn as nn
import torch.nn.functional as F

sentences = [
    "i love machine learning",
    "i love deep learning",
    "machine learning is powerful",
    "deep learning is interesting",
    "i study machine learning"
]

torch.manual_seed(42)

tokenized_sentences = [
    sentence.lower().split()
    for sentence in sentences
]

vocab = sorted(
    set(
        word
        for sentence in tokenized_sentences
        for word in sentence
    )
)

word_to_id = {
    word: idx
    for idx, word in enumerate(vocab)
}

id_to_word = {
    idx: word
    for word, idx in word_to_id.items()
}


vocab_size = len(vocab)     # assume vocal size is 9
embedding_dim = 8


# Last module we
# Model
class SelfAttention(nn.Module):

    def __init__(self, embedding_dim, d_k, d_v):
        super().__init__()

        # learnable parameters
        self.W_Q = nn.Linear(
            embedding_dim,
            d_k,
            bias=False
        )

        self.W_K = nn.Linear(
            embedding_dim,
            d_k,
            bias=False
        )

        self.W_V = nn.Linear(
            embedding_dim,
            d_v,
            bias=False
        )

    def forward(self, X):

        Q = self.W_Q(X)

        K = self.W_K(X)

        V = self.W_V(X)

        scores = Q @ K.transpose(-2, -1)

        d_k = Q.size(-1)

        scores = (
            scores /
            (d_k ** 0.5)
        )

        weights = F.softmax(
            scores,
            dim=-1
        )

        output = weights @ V

        return output, weights


attention = SelfAttention(
    embedding_dim=8,
    d_k=4,
    d_v=4
)

tokens = "i love machine learning".split()

ids = torch.tensor([
    word_to_id[word]
    for word in tokens
])

embedding = nn.Embedding(           # (sentence size, embedding_dim)
    vocab_size,
    embedding_dim
)

X = embedding(ids)

output, weights = attention(X)

print("X:", X.shape)
print("Output:", output.shape)
print("Weights:", weights.shape)

"""
Our attention currently allows:

i → i
i → love
i → machine
i → learning

etc.

That's called full self-attention.
For understanding the mechanism, that's perfect.
But when we build a language model that predicts the next word, we don't want the model cheating.

For example:
    Input:
        i love machine
    When predicting the next word, it shouldn't be able to look at:
        learning

because that's the answer we're trying to predict.
So we need something called a:
    Causal Mask

Suppose:
    i love machine learning

The allowed attention should look like:
                 i    love   machine learning
    i           ✓
    love        ✓      ✓
    machine     ✓      ✓       ✓
    learning    ✓      ✓       ✓       ✓

when those future tokens shouldn't be visible.

Mathematically, we create a mask:
    [
     [0, -∞, -∞, -∞],
     [0,  0, -∞, -∞],
     [0,  0,  0, -∞],
     [0,  0,  0,  0]
    ]

Then add it to the attention scores before softmax.
The -∞ positions become effectively zero probability after softmax.

i study machine learning
    →
    i → study
    i study → machine
    i study machine → learning
"""

X = []
Y = []

for sentence in tokenized_sentences:

    ids = [
        word_to_id[word]
        for word in sentence
    ]

    X.append(ids[:-1])
    Y.append(ids[1:])

X = torch.tensor(X)
Y = torch.tensor(Y)


class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        attention_dim
    ):
        super().__init__()

        self.W_Q = nn.Linear(
            embedding_dim,
            attention_dim,
            bias=False
        )

        self.W_K = nn.Linear(
            embedding_dim,
            attention_dim,
            bias=False
        )

        self.W_V = nn.Linear(
            embedding_dim,
            attention_dim,
            bias=False
        )

    def forward(self, X):

        # --------------------------------
        # 1. Create Q, K, V
        # --------------------------------

        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)


        # --------------------------------
        # 2. QK^T
        # --------------------------------

        scores = Q @ K.transpose(-2, -1)

        # --------------------------------
        # 3. Scale
        # --------------------------------

        d_k = Q.size(-1)

        scores = scores / (d_k ** 0.5)


        # --------------------------------
        # 4. Causal mask
        # --------------------------------

        sequence_length = X.size(1)

        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=X.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # --------------------------------
        # 5. Softmax
        # --------------------------------

        attention_weights = F.softmax(
            scores,
            dim=-1
        )


        # --------------------------------
        # 6. Weighted sum
        # --------------------------------

        output = attention_weights @ V


        return output, attention_weights

"""
mask = torch.tril(
    torch.ones(3, 3)
)

which gives:
    1 0 0
    1 1 0
    1 1 1

Meaning:
             i    love   machine
i            ✓
love         ✓      ✓
machine      ✓      ✓       ✓

Position 1
    i
    
    can only see:
        i

Position 2
    love
    
    can see:
        i
        love

Position 3
    machine
    
    can see:
        i
        love
        machine

It cannot see the future.

We do:
scores.masked_fill(
    mask == 0,
    float("-inf")
)

Suppose our scores are:
    [2.1, 1.5, 3.2]

and we want to hide the last two:
    [2.1, -inf, -inf]

Softmax gives approximately:
    [1.0, 0.0, 0.0]

So the model can't attend to those future positions.
"""

class AttentionLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        attention_dim
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.attention = CausalSelfAttention(
            embedding_dim,
            attention_dim
        )

        self.output_layer = nn.Linear(
            attention_dim,
            vocab_size
        )

    def forward(self, x):

        # Token IDs → embeddings
        x = self.embedding(x)

        # Causal attention
        x, attention_weights = self.attention(x)

        # Attention representation → vocabulary scores
        logits = self.output_layer(x)

        return logits, attention_weights

embedding_dim = 16
attention_dim = 16

model = AttentionLanguageModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    attention_dim=attention_dim
)

print(model)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


epochs = 100

for epoch in range(epochs):

    # -------------------------
    # Forward
    # -------------------------

    logits, attention_weights = model(X)


    # -------------------------
    # Reshape
    # -------------------------

    loss = criterion(
        logits.reshape(-1, vocab_size),
        Y.reshape(-1)
    )


    # -------------------------
    # Backpropagation
    # -------------------------

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    if (epoch + 1) % 10 == 0:

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

        # Last position
        last_logits = logits[:, -1, :]

        # Highest probability
        predicted_id = torch.argmax(
            last_logits,
            dim=-1
        ).item()

    return id_to_word[predicted_id]


print(
    predict_next_word(
        "i love machine"
    )
)

def generate_text(
    start_text,
    num_words
):

    model.eval()

    words = start_text.lower().split()

    for _ in range(num_words):

        next_word = predict_next_word(
            " ".join(words)
        )

        words.append(next_word)

    return " ".join(words)


print(
    generate_text(
        "i love",
        4
    )
)

text = "i love machine"

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

# Our attention matrix:
weights = attention_weights[0]

print(weights)      # [3, 3]

words = text.split()

print(
    "             "
    + " ".join(
        f"{word:>10}"
        for word in words
    )
)

for i, word in enumerate(words):

    row = weights[i]

    print(
        f"{word:>10} "
        + " ".join(
            f"{value.item():10.3f}"
            for value in row
        )
    )

"""
You'll see something like:
                     i       love     machine
         i         1.000      0.000      0.000
      love         0.XXX      0.XXX      0.000
   machine         0.XXX      0.XXX      0.XXX

Notice the zeros.
Those zeros aren't learned.
They are forced by our causal mask.

Causal Self-Attention
              ┌── i
              │
i ────────────┤
              │
              └── ...

love ─────────┼── i
              └── love

machine ──────┼── i
              ├── love
              └── machine

"""