import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pad_sequence


sentences = [
    "i love machine learning",
    "deep learning is powerful",
    "transformers changed natural language processing",
    "attention helps models understand relationships",
    "pytorch makes neural network development easier",
    "language models predict the next token",
    "recurrent networks process sequences step by step",
    "self attention allows every token to interact with previous tokens",
    "large language models learn patterns from massive amounts of text",
    "transformers use attention instead of recurrence to process sequences efficiently"
]

# We'll use two special tokens:
special_tokens = [
    "<PAD>",
    "<UNK>",
    "<EOS>"
]

tokenized = [
    sentence.lower().split()
    for sentence in sentences
]

vocab = sorted(
    set(
        word
        for sentence in tokenized
        for word in sentence
    )
)

vocab = special_tokens + vocab

word_to_id = {
    word: i
    for i, word in enumerate(vocab)
}

id_to_word = {
    i: word
    for word, i in word_to_id.items()
}

vocab_size = len(vocab)

PAD_ID = word_to_id["<PAD>"]
UNK_ID = word_to_id["<UNK>"]
EOS_ID = word_to_id["<EOS>"]

"""
Why padding is necessary ?
    Previously every sentence had the same length.

Now:
    i love machine learning
    deep learning is powerful
    transformers changed natural language processing

have different lengths.
PyTorch tensors need rectangular dimensions.

So we might represent them as:
    i love machine learning <PAD> <PAD>
    deep learning is powerful <PAD> <PAD>
    transformers changed natural language processing

Suppose the maximum sequence length is 10.
Every sequence becomes:
    [10 tokens]

But there is an important problem.
We don't want the model to learn from <PAD>.
So we'll use masking.

Create training examples
    For language modeling, we need:

Input:
    i love machine learning

Target:
    love machine learning <EOS>

"""

encoded_sentences = []

for sentence in tokenized:

    ids = [
        word_to_id.get(
            word,
            UNK_ID
        )
        for word in sentence
    ]

    ids.append(EOS_ID)

    encoded_sentences.append(ids)

max_seq_len = max(
    len(ids)
    for ids in encoded_sentences
)

print("maximum sequence length", max_seq_len)

X_data = []
Y_data = []

for ids in encoded_sentences:

    x = ids[:-1]
    y = ids[1:]

    X_data.append(x)
    Y_data.append(y)


X_data = [
    torch.tensor(x, dtype=torch.long)
    for x in X_data
]

Y_data = [
    torch.tensor(y, dtype=torch.long)
    for y in Y_data
]

X_data = pad_sequence(
    X_data,
    batch_first=True,
    padding_value=PAD_ID
)

Y_data = pad_sequence(
    Y_data,
    batch_first=True,
    padding_value=PAD_ID
)

print("X shape", X_data.shape)
print("Y shape", Y_data.shape)

# for row in X_data:
#     print([
#         id_to_word[token.item()]
#         for token in row
#     ])

"""
We need to tell the model:
    These positions are real tokens; these positions are padding.
"""
padding_mask = (
    X_data != PAD_ID
)


"""
Our Multi-Head Attention needs two masks
There are now two separate concepts.
    Causal mask
        Prevents future tokens from being seen.

    For 4 tokens:
        1 0 0 0
        1 1 0 0
        1 1 1 0
        1 1 1 1

Padding mask
    Prevents:
        <PAD>
    
    from being attended to.

We need both.
"""


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads, max_seq_len):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

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

        self.W_O = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        # Causal mask

        causal_mask = torch.tril(
            torch.ones(
                max_seq_len,
                max_seq_len
            )
        )

        self.register_buffer(
            "causal_mask",
            causal_mask
        )

    def forward(self, x, padding_mask=None):

        batch_size, seq_len, _ = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # -------------------------
        # Attention scores
        # -------------------------

        scores = Q @ K.transpose(-2, -1)

        scores = scores / (
            self.head_dim ** 0.5
        )

        # -------------------------
        # Causal mask
        # -------------------------

        causal = self.causal_mask[
            :seq_len,
            :seq_len
        ]

        scores = scores.masked_fill(
            causal == 0,
            float("-inf")
        )

        # -------------------------
        # Padding mask
        # -------------------------

        if padding_mask is not None:

            key_mask = (
                padding_mask
                .unsqueeze(1)
                .unsqueeze(2)
            )

            scores = scores.masked_fill(
                key_mask == 0,
                float("-inf")
            )

        # -------------------------
        # Softmax
        # -------------------------

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # -------------------------
        # Weighted values
        # -------------------------

        output = attention_weights @ V

        # -------------------------
        # Combine heads
        # -------------------------

        output = output.transpose(
            1,
            2
        )

        output = output.contiguous().view(
            batch_size,
            seq_len,
            self.d_model
        )

        output = self.W_O(output)

        return output, attention_weights


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


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        max_seq_len
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.attention = MultiHeadSelfAttention(
            d_model,
            num_heads,
            max_seq_len
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

        self.ffn = FeedForward(
            d_model,
            d_ff
        )

    def forward(
        self,
        x,
        padding_mask=None
    ):

        # Attention

        normalized_x = self.norm1(x)

        attention_output, weights = \
            self.attention(
                normalized_x,
                padding_mask
            )

        x = x + attention_output

        # FFN

        normalized_x = self.norm2(x)

        ffn_output = self.ffn(
            normalized_x
        )

        x = x + ffn_output

        return x, weights


# Use learned positional embeddings.
# Since we're now making this more GPT-like, let's replace our sinusoidal positional encoding with learned
# positional embeddings.


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

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            d_model
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                max_seq_len
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(
            d_model
        )

        self.output_layer = nn.Linear(
            d_model,
            vocab_size,
            bias=False
        )

    def forward(
        self,
        x,
        padding_mask=None
    ):

        batch_size, seq_len = x.shape

        # Token embedding

        token_emb = self.token_embedding(x)

        # Position IDs

        positions = torch.arange(
            seq_len,
            device=x.device
        )

        position_emb = self.position_embedding(
            positions
        )

        # Add token + position

        x = token_emb + position_emb

        attention_weights = []

        # Transformer blocks

        for block in self.blocks:

            x, weights = block(
                x,
                padding_mask
            )

            attention_weights.append(
                weights
            )

        # Final normalization

        x = self.final_norm(x)

        # Vocabulary logits

        logits = self.output_layer(x)

        return logits, attention_weights


model = GPTModel(
    vocab_size=vocab_size,
    d_model=64,
    num_heads=4,
    d_ff=256,
    num_layers=4,
    max_seq_len=max_seq_len
)


"""
This is extremely important.
    We don't want:
        <PAD>  to contribute to the loss.

Normally:
nn.CrossEntropyLoss() would calculate loss for every target.

Now:
    real token   → contributes to loss
    <PAD>        → ignored


"""
criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_ID
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

for epoch in range(1000):

    model.train()

    padding_mask = (
        X_data != PAD_ID
    )

    logits, attention_weights = model(
        X_data,
        padding_mask
    )

    loss = criterion(
        logits.reshape(
            -1,
            vocab_size
        ),
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


def encode_text(text):

    words = text.lower().split()

    ids = [
        word_to_id.get(
            word,
            UNK_ID
        )
        for word in words
    ]

    return ids


def generate(
    model,
    text,
    max_new_tokens=10,
    temperature=1.0
):

    model.eval()

    ids = encode_text(text)

    x = torch.tensor(
        [ids],
        dtype=torch.long
    )

    for _ in range(max_new_tokens):

        padding_mask = (
            x != PAD_ID
        )

        with torch.no_grad():

            logits, _ = model(
                x,
                padding_mask
            )

        next_logits = logits[:, -1, :]

        next_logits = (
            next_logits / temperature
        )

        probs = F.softmax(
            next_logits,
            dim=-1
        )

        next_id = torch.multinomial(
            probs,
            num_samples=1
        )

        x = torch.cat(
            [x, next_id],
            dim=1
        )

        if next_id.item() == EOS_ID:
            break

    return [
        id_to_word[i]
        for i in x[0].tolist()
    ]


torch.manual_seed(42)

print(
    generate(
        model,
        "i love",
        max_new_tokens=5,
        temperature=0.7
    )
)


print(
    generate(
        model,
        "deep learning",
        max_new_tokens=5,
        temperature=0.7
    )
)


"""
Temperature controls the distribution, but we can additionally restrict the model to the top k candidates.

Suppose:
    machine    0.45
    deep       0.25
    learning   0.10
    data       0.08
    python     0.05

With:
    top_k = 2

we only sample:
    machine
    deep
"""


def top_k_logits(
    logits,
    k
):

    values, indices = torch.topk(
        logits,
        k
    )

    filtered = torch.full_like(
        logits,
        float("-inf")
    )

    filtered.scatter_(
        -1,
        indices,
        values
    )

    return filtered


def generate(
    model,
    text,
    max_new_tokens=10,
    temperature=1.0
):

    model.eval()

    ids = encode_text(text)

    x = torch.tensor(
        [ids],
        dtype=torch.long
    )

    for _ in range(max_new_tokens):

        padding_mask = (
            x != PAD_ID
        )

        with torch.no_grad():

            logits, _ = model(
                x,
                padding_mask
            )

        next_logits = logits[:, -1, :]

        next_logits = top_k_logits(
            next_logits,
            k=5
        )

        probs = F.softmax(
            next_logits,
            dim=-1
        )

        next_id = torch.multinomial(
            probs,
            num_samples=1
        )

        x = torch.cat(
            [x, next_id],
            dim=1
        )

        if next_id.item() == EOS_ID:
            break

    return [
        id_to_word[i]
        for i in x[0].tolist()
    ]


print(
    generate(
        model,
        "i love",
        max_new_tokens=5,
        temperature=0.7
    )
)

"""
Our mini-GPT now looks like this
                         Input
                           │
                           ▼
                      Token IDs
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
       Token Embedding          Position Embedding
             │                           │
             └─────────────┬─────────────┘
                           +
                           │
                           ▼
                  Transformer Block 1
                           │
                           ▼
                  Transformer Block 2
                           │
                           ▼
                  Transformer Block 3
                           │
                           ▼
                  Transformer Block 4
                           │
                           ▼
                       LayerNorm
                           │
                           ▼
                     Linear Head
                           │
                           ▼
                        Logits
                           │
                    Temperature
                           │
                         Top-k
                           │
                           ▼
                      Next Token
                           │
                           └──────────────┐
                                          │
                                          ▼
                                      Input again

That's a decoder-only Transformer language model.

Trace one token through the entire network:

    token ID
       ↓
    embedding vector
       ↓
    + positional embedding
       ↓
    Q, K, V
       ↓
    attention scores
       ↓
    causal mask
       ↓
    softmax
       ↓
    weighted values
       ↓
    multi-head concatenation
       ↓
    W_O
       ↓
    residual
       ↓
    LayerNorm
       ↓
    FFN
       ↓
    residual
       ↓
    next Transformer block
       ↓
    ...
       ↓
    logits
       ↓
    probability of every vocabulary word
"""

