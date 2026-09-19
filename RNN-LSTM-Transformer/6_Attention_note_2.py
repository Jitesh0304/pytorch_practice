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

embedding = nn.Embedding(           # (sentence size, embedding_dim)
    vocab_size,
    embedding_dim
)

sentence = "i love machine learning"

tokens = sentence.split()

token_ids = [
    word_to_id[word]
    for word in tokens
]

x = torch.tensor(
    token_ids,
    dtype=torch.long
)

X = embedding(x)

print(X.shape)

"""
We now have:
             embedding
I          → [8 numbers]
love       → [8 numbers]
machine    → [8 numbers]
learning   → [8 numbers]
"""

d_k = 4     # take a number which can be divisible by embedding_dim
d_v = 4

W_Q = torch.randn(          # (8, 4)
    embedding_dim,
    d_k
)

W_K = torch.randn(          # (8, 4)
    embedding_dim,
    d_k
)

W_V = torch.randn(          # (8, 4)
    embedding_dim,
    d_v
)

Q = X @ W_Q             # (4, 8) @ (8, 4) = (4, 4)
K = X @ W_K             # (4, 8) @ (8, 4) = (4, 4)
V = X @ W_V             # (4, 8) @ (8, 4) = (4, 4)

print("X:", X.shape)
print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)

"""
X
[4 × 8]
 │
 ├── WQ → Q [4 × 4]
 │
 ├── WK → K [4 × 4]
 │
 └── WV → V [4 × 4]
"""


scores = Q @ K.T            # .T  -: Transpose which means swap (row, col) to (col, row)

print(scores.shape)     # (4, 4)
print(scores)

scaled_scores = scores / (d_k ** 0.5)

print(scaled_scores)

attention_weights = F.softmax(
    scaled_scores,
    dim=-1              # dim where the softmax will compute, in this case last dim holds tensors
)

print(attention_weights)        # (4, 4)

print(
    attention_weights.sum(dim=-1)       # sum of all probabilities in a row will be 1
)

words = tokens

print(
    "             "
    + " ".join(f"{word:>10}" for word in words)
)

for i, word in enumerate(words):

    values = attention_weights[i]

    print(
        f"{word:>10} "
        + " ".join(
            f"{v.item():10.3f}"
            for v in values
        )
    )

output = attention_weights @ V      # (4, 4) @ (4, 4) = (4, 4)

print(output.shape)


def self_attention(sentence):

    tokens = sentence.lower().split()

    token_ids = [
        word_to_id[word]
        for word in tokens
    ]

    x = torch.tensor(
        token_ids,
        dtype=torch.long
    )

    X = embedding(x)

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    scores = Q @ K.T

    scaled_scores = (
        scores / (d_k ** 0.5)
    )

    attention_weights = F.softmax(
        scaled_scores,
        dim=-1
    )

    output = attention_weights @ V

    return (
        tokens,
        Q,
        K,
        V,
        attention_weights,
        output
    )


for sentence in sentences:

    tokens, Q, K, V, weights, output = \
        self_attention(sentence)

    print("\n")
    print(sentence)

    print("Attention:")
    print(weights)

"""
The first sentence:
    i love machine learning

    has 4 tokens.

So attention is:
    4 × 4

The second:
    i love deep learning

    also has 4 tokens.

Again:
    4 × 4

But if we had:
    i love machine learning is powerful

we'd get:
    6 × 6

Because every token attends to every token.

For n tokens:
    Attention matrix = n × n
    
This will become very important later when we discuss the computational cost of Transformers.
"""