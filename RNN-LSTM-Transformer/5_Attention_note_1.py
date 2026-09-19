"""
The three concepts are:

    1- Query
           What am I looking for?
    2- Key
           What information do I contain / what do I represent?
    3- Value
           What information should I actually provide?

Think of a dictionary.
    You search with a:
        Query

    The dictionary matches it against:
        Keys

    and retrieves:
        Value

For every token, we create three vectors:
    Q, K, V

For example:

"I" -:
    Q₁, K₁, V₁

"love"
    Q₂, K₂, V₂

Suppose:
    x_t is the embedding.

We learn three matrices:
    W_Q, W_K, W_V

Then:
    Q = X * W_Q
    K = X * W_K
    V = X * W_V

These matrices are learned during training.

Suppose our sentence has four tokens:

    "I love machine learning"

And each token has a 3-dimensional embedding.

For simplicity:

X = [
        [1 0 1],
        [0 2 1],
        [1 1 0],
        [2 1 1]
    ]

Meaning:
    shape -: (4, 3)
        4 tokens
        3-dimensional embeddings

Q = X * W_Q       ( X = (4,3) &  W_Q = (3,2) .... so the Q = (4,2) )
K = X * W_K       ( X = (4,3) &  W_K = (3,2) .... so the K = (4,2) )
V = X * W_V       ( X = (4,3) &  W_V = (3,2) .... so the V = (4,2) )

X
 │
 ├── WQ → Q
 │
 ├── WK → K
 │
 └── WV → V

The most important equation.
    QK.T  (.T = transpose)

Suppose:
    Q = [q1, q2, q3, q4]
    K = [k1, k2, k3, k4]

Then QK.T -:
    [
        [q1.k1  q1.k2  q1.k3  q1.k4],
        [q2.k1  q2.k2  q2.k3  q2.k4],
        [q3.k1  q3.k2  q3.k3  q3.k4],
        [q4.k1  q4.k2  q4.k3  q4.k4]
    ]

This matrix tells us:
    How much does each token attend to every other token?

Suppose we get -:
QK.T = [ [2, 1, 4, 1],
         [1, 3, 2, 1],
         [4, 2, 5, 3],
         [1, 1, 3, 4] ]

Rows are queries.
Columns are keys.

             I    love   machine  learning
          ┌────┬────┬────┬────┐
I         │ 2  │ 1  │ 4  │ 1  │
love      │ 1  │ 3  │ 2  │ 1  │
machine   │ 4  │ 2  │ 5  │ 3  │
learning  │ 1  │ 1  │ 3  │ 4  │
          └────┴────┴────┴────┘

Look at the "learning" row:
    learning → [1, 1, 3, 4]

This says:
    I         → low
    love      → low
    machine   → high
    learning  → highest

That's exactly the kind of relationship we want the model to learn.

Our complete score calculation is:
    S = (QK.T / sqrt(dk))
    A = softmax(S)

where:
    dk     ( it is the dimension of the key vectors. )

if:
    dk = 64
then:
    sqrt(dk) = 8

So QK.T is divided by 8 before softmax.

Why?
    Because dot products can become large when vector dimensions become large.
    Large values fed into softmax can make it extremely peaked:
    [0.00001, 0.00002, 0.99997]

    which can lead to poor gradients.
    Scaling keeps the values in a more manageable range.


Our complete score calculation is:
    S = (QK.T / sqrt(dk))
    A = softmax(S)

Now each row of A sums to 1.

For example:
                 I     love    machine learning
    I         [0.2    0.1      0.6       0.1]
    love      [0.1    0.6      0.2       0.1]
    machine   [0.5    0.1      0.3       0.1]
    learning  [0.05   0.05     0.4       0.5]

    Now these are actual attention weights.

Then:
    Output = AV

Suppose for "learning"
    A4 = [0.05, 0.05, 0.4, 0.5]

and the value vectors are:
    V1, v2, V3, V4

Then
    output_4 = 0.05V1 + 0.05V2 + 0.4V3 + 0.5V4

So "learning" creates a weighted combination of information from all tokens.
That's attention.

Memorize this flow:

             X
             │
       ┌─────┼─────┐
       ↓     ↓     ↓
      WQ    WK    WV
       ↓     ↓     ↓
       Q     K     V
       │     │
       └──┬──┘
          ↓
        QKᵀ
          ↓
     divide √dₖ
          ↓
       softmax
          ↓
   attention weights
          │
          ↓
          × V
          ↓
       OUTPUT

And mathematically -:
    Attention(Q, K, V) = softmax(QKᵀ / √dₖ) * V

"""

import torch
import torch.nn.functional as F

torch.manual_seed(42)

# 4 tokens
# embedding dimension = 3

X = torch.tensor([
    [1.0, 0.0, 1.0],   # I
    [0.0, 2.0, 1.0],   # love
    [1.0, 1.0, 0.0],   # machine
    [2.0, 1.0, 1.0]    # learning
])

d_model = 3     # dimension of model OR embedding size
d_k = 2
d_v = 2

W_Q = torch.randn(d_model, d_k)     # (3, 2)
W_K = torch.randn(d_model, d_k)     # (3, 2)
W_V = torch.randn(d_model, d_v)     # (3, 2)

Q = X @ W_Q         # (4, 3) @ (3, 2) = (4, 2)
K = X @ W_K         # (4, 3) @ (3, 2) = (4, 2)
V = X @ W_V         # (4, 3) @ (3, 2) = (4, 2)

print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)

scores = Q @ K.T        # (4, 2) @ (2, 4) = (4, 4) .... why (2, 4) because K.T which is transpose of K shape

print(scores)

scores = scores / torch.sqrt(           # (4, 4)
    torch.tensor(d_k)
)

attention_weights = F.softmax(          # the last dim of the tensor holds score values
    scores,
    dim=-1
)

print(attention_weights)

print(
    attention_weights.sum(dim=-1)
)

output = attention_weights @ V          # (4, 4) @ (4, 2) = (4, 2)

print(output)
print(output.shape)



# Put everything into one function
def self_attention(X):

    d_model = X.shape[-1]
    d_k = 2
    d_v = 2

    W_Q = torch.randn(
        d_model,
        d_k
    )

    W_K = torch.randn(
        d_model,
        d_k
    )

    W_V = torch.randn(
        d_model,
        d_v
    )

    # 1. Create Q, K, V
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    # 2. Calculate scores
    scores = Q @ K.T

    # 3. Scale
    scores = scores / (d_k ** 0.5)

    # 4. Softmax
    weights = F.softmax(
        scores,
        dim=-1
    )

    # 5. Weighted sum
    output = weights @ V

    return output, weights


output, weights = self_attention(X)

print("Output:")
print(output)

print("\nAttention weights:")
print(weights)

"""
But there's an important problem
    Our W_Q, W_K, and W_V are currently:

    torch.randn(...)

    and they're random.

That means our attention doesn't actually know anything about language.
In a real model, these matrices are learnable parameters.
During training:

prediction
   ↓
loss
   ↓
backpropagation
   ↓
WQ, WK, WV updated

The model learns how to construct useful queries, keys, and values.
This is an extremely important point:
    Attention isn't a manually programmed lookup system. The model learns the representations that make attention useful.

Why is this called self-attention?

Because:
    Q, K, V

all come from the same input sequence.
             X
          /  |  \
         ↓   ↓   ↓
         Q   K   V

So the sentence attends to itself.

For:
    I love machine learning

every token can attend to every token:

            I   love   machine   learning

I           ✓    ✓       ✓          ✓
love        ✓    ✓       ✓          ✓
machine     ✓    ✓       ✓          ✓
learning    ✓    ✓       ✓          ✓


RNN
    x₁ → h₁ → h₂ → h₃ → h₄

To get information from x₁ to h₄:
    x₁ → h₁ → h₂ → h₃ → h₄
    
    Four sequential steps.

Attention
    x₁ ───────────────→ x₄
    x₂ ───────────────→ x₄
    x₃ ───────────────→ x₄
    x₄ ───────────────→ x₄
    
    Direct relationships.

Attention doesn't mean:
    "Every word gets equally important."

It means:
    For each query token, the model computes a different distribution of importance over the tokens.

For example:
    Query = "learning"

    I          0.05
    love       0.05
    machine    0.40
    learning   0.50

But for:
    Query = "machine" .... the distribution could be:
    
    I          0.10
    love       0.20
    machine    0.50
    learning   0.20

Every row of the attention matrix represents a different query.

Attention(Q, K, V) = softmax(QKᵀ / √dₖ) * V
    Take each query, compare it with every key, scale the scores, convert them into probabilities, 
    and use those probabilities to take a weighted combination of the values.
    
    That's all attention is at its core.
"""

