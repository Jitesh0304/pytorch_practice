import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ============================================================
# 1. DATA
# ============================================================
sentences = [
    "i love machine learning",
    "i love deep learning",
    "machine learning is powerful",
    "deep learning is interesting",
    "i study machine learning"
]

# ============================================================
# 2. TOKENIZATION
# ============================================================
tokenized_sentences = [
    sentence.lower().split()
    for sentence in sentences
]

# ============================================================
# 3. VOCABULARY
# ============================================================
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

vocab_size = len(vocab)

print("Vocabulary size:", vocab_size)

# ============================================================
# 4. ENCODE SENTENCES
# ============================================================

encoded_sentences = [
    [word_to_id[word] for word in sentence]
    for sentence in tokenized_sentences
]

print("\nEncoded sentences:")

for sentence in encoded_sentences:
    print(sentence)


# ============================================================
# 5. CREATE INPUT / TARGET
# ============================================================

X = []
Y = []

for sentence in encoded_sentences:
    X.append(sentence[:-1])
    Y.append(sentence[1:])


X = torch.tensor(
    X,
    dtype=torch.long
)

Y = torch.tensor(
    Y,
    dtype=torch.long
)

# LSTM Model
class LSTMLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()

        # Word ID → vector
        self.embedding = nn.Embedding(
            num_embeddings= vocab_size,
            embedding_dim= embedding_dim
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Hidden state → vocabulary scores
        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(self, x):

        # [batch, sequence]
        #       ↓
        # [batch, sequence, embedding]
        x = self.embedding(x)

        # LSTM
        output, (hidden, cell) = self.lstm(x)

        # [batch, sequence, hidden]
        #       ↓
        # [batch, sequence, vocab]
        output = self.fc(output)

        return output


embedding_dim = 16
hidden_size = 32

model = LSTMLanguageModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size
)

"""
Conceptually:

    Vocabulary ID
          ↓
    Embedding
          ↓
    16-dimensional vector
          ↓
    LSTM
          ↓
    32-dimensional hidden state
          ↓
    Linear
          ↓
    9 vocabulary scores
"""

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)

epochs = 1000

for epoch in range(epochs):

    # Forward pass
    output = model(X)

    # output:
    # [5, 3, 9]

    # target:
    # [5, 3]

    loss = criterion(
        output.reshape(-1, vocab_size),
        Y.reshape(-1)
    )

    # Clear previous gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()

    if (epoch + 1) % 100 == 0:

        print(
            f"Epoch {epoch + 1:4d} "
            f"Loss: {loss.item():.4f}"
        )

def predict_next_word(text):

    model.eval()

    words = text.lower().split()

    token_ids = [
        word_to_id[word]
        for word in words
    ]

    x = torch.tensor(
        [token_ids],
        dtype=torch.long
    )

    with torch.no_grad():

        output = model(x)

        # Take the final time step
        last_output = output[:, -1, :]

        # Highest scoring word
        predicted_id = torch.argmax(
            last_output,
            dim=-1
        ).item()

    return id_to_word[predicted_id]

print(
    predict_next_word("i love machine")
)

def generate_text(start_text, num_words):

    model.eval()

    words = start_text.lower().split()

    for _ in range(num_words):

        next_word = predict_next_word(
            " ".join(words)
        )

        words.append(next_word)

    return " ".join(words)

print(
    generate_text("i love", 5)
)

"""
Let's make prediction probabilities visible
argmax() only gives us the most likely word.

But the model actually produces scores for every word.
"""

def predict_with_probabilities(text):

    model.eval()

    words = text.lower().split()

    token_ids = [
        word_to_id[word]
        for word in words
    ]

    x = torch.tensor(
        [token_ids],
        dtype=torch.long
    )

    with torch.no_grad():

        output = model(x)

        logits = output[:, -1, :]

        probabilities = torch.softmax(
            logits,
            dim=-1
        )[0]

    results = []

    for idx, probability in enumerate(probabilities):

        results.append(
            (
                id_to_word[idx],
                probability.item()
            )
        )

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results


results = predict_with_probabilities(
    "i love machine"
)

for word, probability in results:

    print(
        f"{word:12s} "
        f"{probability:.4f}"
    )

model.eval()

x = X[0].unsqueeze(0)

with torch.no_grad():

    embedded = model.embedding(x)

    output, (hidden, cell) = model.lstm(
        embedded
    )

print("Input shape:")
print(x.shape)          # [batch, sequence]

print("\nEmbedding shape:")
print(embedded.shape)   # [batch, sequence, embedding]

print("\nOutput shape:")
print(output.shape)     # [batch, sequence, hidden]

print("\nHidden shape:")
print(hidden.shape)     # [batch, 1, hidden]

print("\nCell shape:")
print(cell.shape)       # [batch, 1, hidden]

# after linear [batch, sequence, vocabulary]


"""
Our implementation:

xₜ + hₜ₋₁
      ↓
 ┌───────────────┐
 │ forget gate   │
 │ input gate    │
 │ candidate     │
 │ output gate   │
 └───────────────┘
      ↓
   Cₜ, hₜ


PyTorch:

xₜ + hₜ₋₁
      ↓
   nn.LSTM
      ↓
   Cₜ, hₜ
"""

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
        f"{text:25s} → "
        f"{predict_next_word(text)}"
    )
