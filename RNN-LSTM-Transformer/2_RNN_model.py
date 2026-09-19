import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)

sentences = [
    "i love machine learning",
    "i love deep learning",
    "machine learning is powerful",
    "deep learning is interesting",
    "i study machine learning"
]

# make list of words of each sentence
tokenized_sentences = [
    sentence.lower().split()
    for sentence in sentences
]

# all unique words
vocab = sorted(set(
    word
    for sentence in tokenized_sentences
    for word in sentence
))

# word to ID
word_to_id = {
    word: i
    for i, word in enumerate(vocab)
}

# ID to word
id_to_word = {
    i: word
    for word, i in word_to_id.items()
}

# tokenization (convert string sentences to list of IDs)
encoded_sentences = [
    [word_to_id[word] for word in sentence]
    for sentence in tokenized_sentences
]

# prepare dataset for training
X = []  # keep all tokens excluding the last token
Y = []  # keep all tokens excluding the first token
for sentence in encoded_sentences:
    x = sentence[:-1]
    y = sentence[1:]

    X.append(x)
    Y.append(y)

# convert list of integer to tenson
X = torch.tensor(X, dtype=torch.long, device=device)
Y = torch.tensor(Y, dtype=torch.long, device=device)

print(X.shape)
print(Y.shape)


# RNN Model
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()

        # If the word "apple" is token index 5, it will get the exact same vector every single time
        # it appears in your text during that forward pass. After Backpropagation the vectors change and so on
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim,
                                      device=device)

        self.rnn = nn.RNN(input_size=embedding_dim,  # The number of features in each word vector
                          hidden_size=hidden_size,
                          # The number of features in the RNN's internal memory (controls how much it can remember)
                          batch_first=True,
                          # Tells the RNN that data is shaped as (Batch Size, Sequence Length, Features)
                          device=device)

        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size, device=device)

    def forward(self, x):
        # Word IDs → embeddings
        x = self.embedding(x)

        # Embeddings → hidden states
        output, hidden = self.rnn(x)

        # Hidden states → vocabulary scores
        output = self.fc(output)

        return output


"""
"i"
 ↓
ID = 1
 ↓
Embedding
 ↓
[0.23, -0.41, 0.71, ...]
 ↓
RNN
 ↓
hidden state
 ↓
Linear
 ↓
scores for every word

Suppose vocabulary size = 6.

The final layer produces:

[
  score(machine),
  score(i),
  score(love),
  score(learning),
  score(is),
  score(powerful)
]
"""

vocab_size = len(vocab)  # total unique words

embedding_dim = 16  # size of embedding vector
hidden_size = 32  # rnn previous word memory size

model = RNNLanguageModel(
    vocab_size=vocab_size, embedding_dim=embedding_dim,
    hidden_size=hidden_size
)

"""
Vocabulary
    ↓
Embedding
9 × 16
    ↓
RNN
16 → 32
    ↓
Linear
32 → 9
"""

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(
    model.parameters(),  # update the parameter
    lr=0.01
)

epochs = 1000


def train(epochs):
    for epoch in range(epochs):

        # Forward pass
        output = model(X)
        model.to(device)

        # output shape:
        # [batch, sequence, vocab_size]

        # Reshape for CrossEntropyLoss
        loss = criterion(
            output.reshape(-1, vocab_size),
            Y.reshape(-1)
        )

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1}, "
                f"Loss: {loss.item():.4f}"
            )
    return model


model = train(epochs)
"""
Input             Target

i                 love
i love            machine
i love machine    learning
"""


# convert text to tensor
def encode(text: str):
    words = text.lower().split()

    return torch.tensor(
        [[word_to_id[word] for word in words]],
        dtype=torch.long,
        device=device
    )


# convert word id to text
def decode(ids: list[int]):
    return [
        id_to_word[int(i)]
        for i in ids
    ]


def predict_next_word(sentence: str):
    model.eval()

    x = encode(sentence)

    with torch.no_grad():
        output = model(x)

        # Get prediction from final timestep, only last word prediction
        last_output = output[:, -1, :]

        # maximum score word
        predicted_id = torch.argmax(
            last_output,
            dim=-1
        ).item()

        predicted_word = id_to_word[predicted_id]

    print(predicted_word)


predict_next_word("i love machine")


def generate_text(start_text, num_words):
    model.eval()

    words = start_text.lower().split()

    # predict next given number of words
    for _ in range(num_words):
        x = torch.tensor(
            [[word_to_id[word] for word in words]],
            dtype=torch.long,
            device=device
        )

        with torch.no_grad():
            output = model(x)

            # Get prediction from final timestep, only last word prediction
            last_output = output[:, -1, :]

            # maximum score word
            predicted_id = torch.argmax(
                last_output,
                dim=-1
            ).item()

        predicted_word = id_to_word[predicted_id]

        words.append(predicted_word)

    return " ".join(words)


print(generate_text("i love", 5))

# Our RNN again

"""
x₁ → h₁ → h₂ → h₃ → y₃
      ↑     ↑     ↑
     Wₕ    Wₕ    Wₕ

The important thing is that the same Wₕ is reused at every time step.



Why normal backpropagation isn't enough ?

x₁ ──→ h₁ ──→ h₂ ──→ h₃ ──→ y₃
         ↑       ↑       ↑
        Wₕ      Wₕ      Wₕ

Because h₂ depends on h₁, and h₃ depends on h₂, the gradient has to travel backward through time.
Hence:
    Backpropagation Through Time

          ┌───────────────┐
x₁ ──────→│               │
h₀ ──────→│      RNN      │──→ h₁
          │               │
          └───────────────┘
                   │
                   ▼
          ┌───────────────┐
x₂ ──────→│      RNN      │──→ h₂
h₁ ──────→│               │
          └───────────────┘
                   │
                   ▼
          ┌───────────────┐
x₃ ──────→│      RNN      │──→ h₃
h₂ ──────→│               │
          └───────────────┘
                   │
                   ▼
                   L

The loss can influence Wₕ through:

Path 1:
    Wₕ → h₃ → L

Path 2:
    Wₕ → h₂ → h₃ → L

Path 3:
    Wₕ → h₁ → h₂ → h₃ → L

The gradient gets multiplied repeatedly. This is the vanishing-gradient problem

x₁ → h₁ → h₂ → h₃ → h₄ → ... → h₂₀ → output

To learn something about x₁, the gradient must travel:

output
 ↓
h₂₀
 ↓
h₁₉
 ↓
h₁₈
 ↓
...
 ↓
h₁

At every step, the gradient gets multiplied by something like:

    Wₕ(1 - hₜ^2)

Eventually:
    ≈ 0
The network effectively says:
    "I can't learn from that old information."


Exploding gradients
    The opposite can happen.

Then:

1.5¹  = 1.5
1.5²  = 2.25
1.5³  = 3.375
...
1.5²⁰ ≈ 3325

The gradient becomes enormous.
    This is the:
        Exploding gradient problem

import numpy as np

for value in [0.1, 0.5, 0.9, 1.1, 1.5]:

    gradient = value ** 20

    print(
        f"value={value}, "
        f"gradient={gradient}"
    )

You'll see approximately:

value=0.1  → 0.00000000000000000001
value=0.5  → 0.000000953674
value=0.9  → 0.121576
value=1.1  → 6.7275
value=1.5  → 3325.26

That's the fundamental problem.

If the repeated multiplication factor is:
    < 1
    the gradient tends toward zero.

If:
    > 1
    it can explode.

Why this matters for language
Consider:
    "The boy who lived in the house near the river that flooded last year was happy."

The model needs to connect:
    boy ─────────────────────────────→ was

But if the sequence is long:
    boy → who → lived → in → the → house → near → ...
                                                 ↓
                                                was

The gradient has to travel through many recurrent steps.
Vanilla RNNs struggle with this.

That's the fundamental problem.

If the repeated multiplication factor is:

< 1

the gradient tends toward zero.

If:

> 1

it can explode.

13. Why this matters for language
Consider:

"The boy who lived in the house near the river that flooded last year was happy."

The model needs to connect:

boy ─────────────────────────────→ was

But if the sequence is long:

boy → who → lived → in → the → house → near → ...
                                             ↓
                                            was

The gradient has to travel through many recurrent steps.

Vanilla RNNs struggle with this.

Vanilla RNN
Its memory is:
    hₜ

Its weakness:
    long-term dependencies
           ↓
    vanishing/exploding gradients


This leads us to LSTM
    Researchers developed LSTM — Long Short-Term Memory networks to address this problem.

Instead of having only:
    hidden state

LSTM introduces:
    cell state and gates:

    1. Forget gate
    2. Input gate
    3. Output gate

Conceptually:
                 ┌───────────────┐
                 │               │
hₜ₋₁ ───────────→ │               │
                 │     LSTM      │──→ hₜ
Cₜ₋₁ ───────────→ │               │
                 │               │
xₜ ─────────────→ │               │
                 └───────────────┘
                        │
                        ▼
                       Cₜ

The clever idea is that the cell state provides a much better path for information and gradients to travel through time.

LSTM
    Adds:
    
    cell state
    +
    gates
    
    to improve long-term memory.

"""