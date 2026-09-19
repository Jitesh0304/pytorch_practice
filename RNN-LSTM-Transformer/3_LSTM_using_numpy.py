import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
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


embedding_dim = 16
hidden_size = 32
vocab_size = len(vocab)

embedding = nn.Embedding(
    vocab_size,
    embedding_dim,
    device=device
)


"""
We have four transformations:
    Forget
    Input
    Candidate
    Output

Each needs weights.

We could create:
    Wf
    Wi
    Wc
    Wo
    
We combine all four into one matrix:
"""

W = torch.randn(
    4 * hidden_size,        # combine all 4 weight matrix, shape [128, 48]
    embedding_dim + hidden_size,
    device=device
) * 0.1

b = torch.zeros(
    4 * hidden_size, device=device         # combine all 4 bias matrix, shape [128]
)

W.requires_grad_()
b.requires_grad_()


# build LSTM cell
def lstm_step(x, h_prev, c_prev):

    # Combine input and previous hidden state
    combined = torch.cat(
        [x, h_prev]                     # x : 16 and h : 32 , so the torch.cat will be 48 shape
    )

    # Calculate all four transformations
    gates = W @ combined + b

    # Split into four parts
    f, i, c_candidate, o = torch.chunk(
        gates,
        4
    )

    # Apply activations
    f = torch.sigmoid(f)
    i = torch.sigmoid(i)
    c_candidate = torch.tanh(c_candidate)
    o = torch.sigmoid(o)

    # Update cell state
    c = (
        f * c_prev
        +
        i * c_candidate
    )

    # Update hidden state
    h = (
        o * torch.tanh(c)
    )

    return h, c
"""
h              hidden state
c              cell state
f              forget gate
i              input gate
c_candidate    candidate memory
o              output gate
"""

word_id = word_to_id["i"]

x = embedding(
    torch.tensor(word_id, device=device)
)

# Initialize memory:
h = torch.zeros(hidden_size, device=device)
c = torch.zeros(hidden_size, device=device)

h, c = lstm_step(
    x,
    h,
    c
)

print(h.shape)
print(c.shape)

"""
We now have:
    xₜ
     │
     ▼
    LSTM
     │
     ├──→ hₜ
     │
     └──→ Cₜ
"""

# Let's make a forward function.
def lstm_forward(sequence):

    h = torch.zeros(hidden_size, device=device)
    c = torch.zeros(hidden_size, device=device)

    hidden_states = []

    for token_id in sequence:

        x = embedding(token_id)

        h, c = lstm_step(
            x,
            h,
            c
        )

        hidden_states.append(h)

    return torch.stack(hidden_states)


sequence = X[0]

hidden_states = lstm_forward(sequence)

print(hidden_states.shape)

"""
We have:
    i       → h₁
    love    → h₂
    machine → h₃
"""

# Add the output layer, We want next-word prediction.
Wy = torch.randn(
    vocab_size,
    hidden_size, device=device
) * 0.1

by = torch.zeros(vocab_size, device=device)

Wy.requires_grad_()
by.requires_grad_()

# Then modify our forward pass:
def lstm_forward(sequence):

    h = torch.zeros(hidden_size, device=device)
    c = torch.zeros(hidden_size, device=device)

    outputs = []

    for token_id in sequence:

        x = embedding(token_id)

        h, c = lstm_step(
            x,
            h,
            c
        )

        logits = Wy @ h + by

        outputs.append(logits)

    return torch.stack(outputs)


output = lstm_forward(X[0])
print(output.shape)


loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    [
        W,
        b,
        Wy,
        by,
        *embedding.parameters()
    ],
    lr=0.01
)

# Training
for epoch in range(1000):

    total_loss = 0

    for x, y in zip(X, Y):

        output = lstm_forward(x)

        loss = loss_fn(
            output,
            y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 100 == 0:

        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {total_loss:.4f}"
        )


def lstm_step(x, h_prev, c_prev):

    combined = torch.cat([x, h_prev])

    gates = W @ combined + b

    f, i, c_candidate, o = torch.chunk(gates, 4)

    f = torch.sigmoid(f)
    i = torch.sigmoid(i)
    c_candidate = torch.tanh(c_candidate)
    o = torch.sigmoid(o)

    c = f * c_prev + i * c_candidate

    h = o * torch.tanh(c)

    return h, c, f, i, c_candidate, o


def lstm_forward(sequence, return_gates=False):

    h = torch.zeros(hidden_size, device= device)
    c = torch.zeros(hidden_size, device= device)

    outputs = []

    gate_history = []

    for token_id in sequence:

        x = embedding(token_id)

        h, c, f, i, c_candidate, o = lstm_step(
            x,
            h,
            c
        )

        logits = Wy @ h + by

        outputs.append(logits)

        if return_gates:
            gate_history.append({
                "forget": f.detach(),
                "input": i.detach(),
                "candidate": c_candidate.detach(),
                "output": o.detach(),
                "cell": c.detach(),
                "hidden": h.detach()
            })

    outputs = torch.stack(outputs)

    if return_gates:
        return outputs, gate_history

    return outputs


# Train the LSTM
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    [
        W,
        b,
        Wy,
        by,
        *embedding.parameters()
    ],
    lr=0.01
)

for epoch in range(1000):

    total_loss = 0

    for x, y in zip(X, Y):

        output = lstm_forward(x)

        loss = loss_fn(
            output,
            y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 100 == 0:

        print(
            f"Epoch {epoch + 1}: "
            f"loss={total_loss:.4f}"
        )

sequence = X[0]

print(sequence)

output, gates = lstm_forward(
    sequence,
    return_gates=True
)

for t, gate in enumerate(gates):

    word = id_to_word[int(sequence[t])]

    print(f"\nTime step {t + 1}")
    print("Word:", word)

    print(
        "Forget:",
        gate["forget"].mean().item()
    )

    print(
        "Input:",
        gate["input"].mean().item()
    )

    print(
        "Output:",
        gate["output"].mean().item()
    )


def predict_next_word(text):

    words = text.lower().split()

    token_ids = [
        word_to_id[word]
        for word in words
    ]

    h = torch.zeros(hidden_size, device= device)
    c = torch.zeros(hidden_size, device= device)

    for token_id in token_ids:

        x = embedding(
            torch.tensor(token_id, device= device)
        )

        h, c, _, _, _, _ = lstm_step(
            x,
            h,
            c
        )

    logits = Wy @ h + by

    predicted_id = torch.argmax(
        logits
    ).item()

    return id_to_word[predicted_id]

print(
    predict_next_word("i love machine")
)


def generate_text(start_text, num_words):

    words = start_text.lower().split()

    for _ in range(num_words):

        next_word = predict_next_word(
            " ".join(words)
        )

        words.append(next_word)

    return " ".join(words)


print(
    generate_text("i love", 4)
)

"""
RNN
 ↓
LSTM

LSTM is significantly better at maintaining information over long sequences.
But it still has a fundamental characteristic:
    x₁ → x₂ → x₃ → x₄ → x₅ → ...

The sequence must be processed step by step.
You can't fully process:
    x₁
    x₂
    x₃
    x₄

independently because:
    h2
    
  depends on:
    h1

and:
    h3

  depends on:
    h2

etc.
"""















"""
LSTM has:
             ┌───────────────┐
             │               │
xₜ ─────────→│     LSTM      │──→ hₜ
             │               │
hₜ₋₁ ───────→│               │
             │               │
Cₜ₋₁ ───────→│               │──→ Cₜ
             └───────────────┘

Two states:
    - hidden state
    - cell state
    
The cell state is the important addition.
    
Imagine you're reading a book.
Your cell state is a notebook.
At every word, you have three decisions:

Forget gate
    "Is something in my notebook no longer relevant?"

Input gate
    "Should I write this new information into my notebook?"

Output gate
    "What information from my notebook should I use right now?"

So:
              LSTM
       ┌─────────────────┐
       │                 │
       │  📖 MEMORY      │
       │                 │
       └─────────────────┘
          ↑     ↑     ↓
       forget  write  read
"""



"""
vanilla RNN:
    The problem was long sequences.
        x₁ → h₁ → h₂ → h₃ → ... → h₁₀₀

During backpropagation, gradients repeatedly get multiplied.

They can:
    become extremely small → vanishing gradient
    become extremely large → exploding gradient
    So we want a better memory mechanism.

LSTM introduces a separate cell state:
                 Cell state
Cₜ₋₁ ─────────────────────────→ Cₜ
             ↑
             │
           gates
             │
xₜ ───────→ LSTM ───────→ hₜ
              ↑
             hₜ₋₁

Think of:

C = long-term memory
h = current/short-term representation
The key innovation is that the cell state can carry information through many time steps with relatively little modification.

The three gates
An LSTM has three major gates:

              ┌─────────────┐
xₜ ──────────→│             │
hₜ₋₁ ────────→│    LSTM     │
              │             │
              └─────────────┘
                    │
        ┌───────────┼───────────┐
        ↓           ↓           ↓
     Forget       Input       Output
      gate        gate         gate
        │           │           │
        ↓           ↓           ↓
       fₜ          iₜ          oₜ

They answer three questions:

Forget gate
    What old information should I forget?

Input gate
    What new information should I store?

Output gate
    What information should I expose as the current hidden state?

"""