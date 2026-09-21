"""
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) * V

During training, we process the whole sequence.
During generation, however, we generate one token at a time.
That's where the KV cache becomes extremely important.

What happens without KV cache?
Suppose the model generates:
        The

    Then:
        The cat

    Then:
        The cat sat

    Then:
        The cat sat on

    At every step, our naive implementation sends the entire sequence back through the Transformer.

This means we're repeatedly computing the same keys and values.

The  → Q₁ K₁ V₁
cat  → Q₂ K₂ V₂
sat  → Q₃ K₃ V₃

When generating the next token:
    The cat sat on

we don't need to recalculate:
    K₁ V₁
    K₂ V₂
    K₃ V₃

Those haven't changed.
We only need:
    Q₄ K₄ V₄

and then reuse:
    K₁ K₂ K₃
    V₁ V₂ V₃

This is the KV cache.

Without cache:
             Full sequence
                   │
                   ▼
              Q K V
                   │
                   ▼
              Attention
Every generation step repeats this

With cache:
    Previous tokens
          │
          ▼
     ┌──────────────┐
     │Cached K and V│
     └───────┬──────┘
             │
             │
    New token
       │
       ▼
    New Q K V
       │
       ├──────────────┐
       │              │
       ▼              ▼
    New Q        Cached K,V
       │              │
       └──────┬───────┘
              ▼
          Attention

Why only K and V?
    Remember:
        Attention(Q, K, V) = softmax(QKᵀ / √dₖ) * V

The new token needs to ask:
    "Which previous tokens are relevant to me?"

That's the new Q.
    But previous tokens' K, V are unchanged.

    So we cache K, V.

    but normally calculate the new Q at each generation step.

Why We Use KV Caching During Inference ?
    During training, the model looks at the entire sequence of tokens all at once. During inference, the model
    genrates tokens one by one in a loop, where each new token depends on the previous tokens.

    LLMs generate text auto-regressively (one token after another)

    To predict token #50, the model needs to calculate attention scores using tokens 1 through 49. Without a cache,
    when it moves on to predict token #51, it would have to recompute the Keys (K) and Values (V)
    for tokens 1 through 50 all over again.

Why We Do Not Use KV Caching During Training ?
    Training is Fully Parallelized (The "Causal Mask" Trick)

    During training, we already know the entire ground-truth sequence (both the prompt and the target answer).
    Instead of feeding the model one word at a time, we feed the entire sequence into the GPU at once.
    To make sure token #5 doesn’t cheat by looking at token #6, we apply a mathematical causal attention mask
    (a lower-triangular matrix) that blanks out future tokens. Because the whole sequence is processed in a
    single forward pass, there are no sequential "future steps" to cache anything for.

"""

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

padding_mask = (
    X_data != PAD_ID
)



class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads
    ):
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

    def forward(
        self,
        x,
        padding_mask=None,
        past_k=None,
        past_v=None,
        use_cache=False
    ):
        """
        x:
            [B, T, D]

        past_k:
            [B, H, T_past, head_dim]

        past_v:
            [B, H, T_past, head_dim]

        Returns:
            output
            present_k
            present_v
        """

        batch_size, seq_len, _ = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # --------------------------------------------------
        # [B, T, D]
        # ->
        # [B, H, T, head_dim]
        # --------------------------------------------------

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

        # Then the current token(s) produce:
        #     K_current = [B, H, query_len, head_dim]
        #     V_current = [B, H, query_len, head_dim]
        #
        # K = torch.cat([past_k, K_current], dim=2)
        # V = torch.cat([past_v, V_current], dim=2)
        #
        # Therefore:
        #     K = [B, H, past_len + query_len, head_dim]
        #     V = [B, H, past_len + query_len, head_dim]
        #
        # key_len = past_len + query_len

        past_len = 0
        if past_k is not None:                  # past_k = [B, H, past_len, head_dim]
            past_len = past_k.size(2)
            K = torch.cat(
                [past_k, K],
                dim=2
            )

        if past_v is not None:                  # past_v = [B, H, past_len, head_dim]
            V = torch.cat(
                [past_v, V],
                dim=2
            )

        # --------------------------------------------------
        # Attention scores
        #
        # Q: [B, H, query_len, head_dim]
        # K: [B, H, key_len, head_dim]
        #
        # scores:
        # [B, H, query_len, key_len]
        # --------------------------------------------------
        #
        # B = 2, H = 4, past_len = 10, query_len = 1, head_dim = 8
        # Q = [2, 4, 1, 8]
        # K = [2, 4, 11, 8]
        # key_len = 10 + 1 = 11
        # K.transpose(-2, -1)     [2, 4, 8, 11]
        # Q @ Kᵀ
        #     [2, 4, 1, 8]
        #         @
        #     [2, 4, 8, 11]
        #         ↓
        #         ↓
        #     [2, 4, 1, 11]


        scores = Q @ K.transpose(-2, -1)        # [B, H, query_len, key_len]  @ [B, H, key_len, query_len]

        scores = scores / (                     # [B, heads, query_len, key_len]
            self.head_dim ** 0.5
        )

        # --------------------------------------------------
        # Causal masking
        #
        # Important:
        #
        # During training:
        #       Q = [token1, token2, token3]
        #       K = [token1, token2, token3]
        #
        #       K
        #       1  2  3
        # Q 1   ✓  ✗  ✗
        #   2   ✓  ✓  ✗
        #   3   ✓  ✓  ✓
        #
        # During decode:
        #       Q length = 1
        #       K length = past + 1
        #
        # For decode, no causal mask is actually needed
        # because the new token is the only query.
        # --------------------------------------------------

        query_len = Q.size(2)       # [B, H, T, head_dim]
        key_len = K.size(2)         # [B, H, T, head_dim]

        if query_len > 1:

            past_len = key_len - query_len

            causal_mask = torch.tril(
                torch.ones(
                    query_len,
                    key_len,
                    device=x.device,
                    dtype=torch.bool
                ),
                diagonal=past_len
            )

            # scores = scores.masked_fill(
            #     ~causal_mask,
            #     torch.finfo(scores.dtype).min
            # )
            scores = scores.masked_fill(
                ~causal_mask[
                    None,
                    None,
                    :,
                    :
                ],
                torch.finfo(
                    scores.dtype
                ).min
            )

        # Why query_len > 1?
        #
        # Training / prefill
        # Suppose your input is:
        #       "The cat is sleeping"
        #
        # You process all tokens simultaneously:
        #   Q = Q1 Q2 Q3 Q4
        #   K = K1 K2 K3 K4
        #   V = V1 V2 V3 V4

        # Therefore:
        #       query_len = 4         key_len = 4
        #              Keys
        #            1   2   3   4
        #         ┌───────────────
        # Q1      │ ✓   X   X   X
        # Q2      │ ✓   ✓   X   X
        # Q3      │ ✓   ✓   ✓   X
        # Q4      │ ✓   ✓   ✓   ✓
        #
        # You need the causal mask because:
        #       Q1 cannot see K2, K3, K4
        #       Q2 cannot see K3, K4
        #       Q3 cannot see K4
        #
        # So:
        #     if query_len > 1:
        #         causal_mask = ...
        #
        # It is basically saying:
        #     "If I'm processing multiple new tokens at once,
        #     I need to explicitly prevent each token from seeing future tokens."

        # --------------------------------------------------
        # PADDING MASK
        # --------------------------------------------------

        if padding_mask is not None:

            if padding_mask.size(1) != key_len:

                raise ValueError(
                    "padding_mask length must "
                    "match the total key length. "
                    f"Got padding_mask={padding_mask.size(1)}, "
                    f"key_len={key_len}"
                )

            # padding_mask:
            #
            # [B, total_len]
            #
            # True  = valid token
            # False = PAD
            #
            scores = scores.masked_fill(
                ~padding_mask[:, None, None, :],
                torch.finfo(scores.dtype).min
            )

        # --------------------------------------------------
        # Attention
        # --------------------------------------------------

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # --------------------------------------------------
        # Weighted values
        # --------------------------------------------------

        output = attention_weights @ V

        # [B, H, T, head_dim]
        #
        # ->
        #
        # [B, T, D]
        # --------------------------------------------------

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

        if use_cache:

            present_k = K
            present_v = V

        else:

            present_k = None
            present_v = None

        return (
            output,
            attention_weights,
            present_k,
            present_v
        )



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
        padding_mask=None,
        past_k=None,
        past_v=None,
        use_cache=False
    ):

        # Attention

        normalized_x = self.norm1(x)

        (
            attention_output,
            attention_weights,
            present_k,
            present_v
        ) = self.attention(
            normalized_x,
            padding_mask=padding_mask,
            past_k=past_k,
            past_v=past_v,
            use_cache=use_cache
        )
        x = x + attention_output

        # FFN

        normalized_x = self.norm2(x)

        ffn_output = self.ffn(
            normalized_x
        )

        x = x + ffn_output

        return (
            x,
            attention_weights,
            present_k,
            present_v
        )


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

        self.max_seq_len = max_seq_len

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
        padding_mask=None,
        past_key_values=None,
        use_cache=False
    ):

        """
        x:
            [B, T]

        past_key_values:
            list of tuples:

            [
                (K_layer0, V_layer0),
                (K_layer1, V_layer1),
                ...
            ]

        Returns:

            logits:
                [B, T, vocab_size]

            presents:
                updated KV cache
        """

        batch_size, seq_len = x.shape

        # --------------------------------------------------
        # Determine how many tokens are already cached
        # --------------------------------------------------

        if past_key_values is None:

            past_len = 0

        else:

            # All layers have same sequence length
            past_len = past_key_values[0][0].size(2)

        current_positions = torch.arange(
            past_len,
            past_len + seq_len,
            device=x.device
        )

        if (
            current_positions[-1]
            >= self.max_seq_len
        ):
            raise ValueError(
                "Sequence length exceeds "
                "max_seq_len"
            )

        # Token embedding

        token_emb = self.token_embedding(x)

        # --------------------------------------------------
        # Position embedding
        # --------------------------------------------------

        position_emb = self.position_embedding(
            current_positions
        )

        position_emb = position_emb.unsqueeze(0)

        # Add token + position

        x = token_emb + position_emb


        # --------------------------------------------------
        # Transformer layers
        # --------------------------------------------------

        attention_weights = []

        presents = []

        # Transformer blocks

        for layer_idx, block in enumerate(
                self.blocks
        ):
            if past_key_values is None:

                past_k = None
                past_v = None

            else:

                past_k, past_v = (
                    past_key_values[layer_idx]
                )

            # ----------------------------------------------
            # Forward block
            # ----------------------------------------------

            (
                x,
                weights,
                present_k,
                present_v
            ) = block(
                x,
                padding_mask=padding_mask,
                past_k=past_k,
                past_v=past_v,
                use_cache=use_cache
            )

            attention_weights.append(
                weights
            )

            if use_cache:

                presents.append(
                    (
                        present_k,
                        present_v
                    )
                )

        # --------------------------------------------------
        # Final normalization
        # --------------------------------------------------

        x = self.final_norm(x)

        # --------------------------------------------------
        # LM head ( Vocabulary logits )
        # --------------------------------------------------

        logits = self.output_layer(x)

        if use_cache:

            return (
                logits,
                attention_weights,
                presents
            )

        return (
            logits,
            attention_weights
        )


model = GPTModel(
    vocab_size=vocab_size,
    d_model=64,
    num_heads=4,
    d_ff=256,
    num_layers=4,
    # max_seq_len=max_seq_len       # change the max_seq_length
    max_seq_len=max_seq_len     # training sequence length and maximum generation length are different concepts.
)


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

    # Notice that we don't use the KV cache during training. That's intentional.
    logits, attention_weights = model(
        X_data, padding_mask=padding_mask
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



# We will use KV cache during inference.
@torch.no_grad()
def generate(
    model,
    text,
    max_new_tokens=10,
    temperature=1.0
):

    model.eval()

    ids = encode_text(text)

    if len(ids) == 0:
        raise ValueError(
            "Input text cannot be empty"
        )

    device = next(
        model.parameters()
    ).device

    # --------------------------------------------------
    # Initial prompt
    # --------------------------------------------------

    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device
    )

    # --------------------------------------------------
    # Prompt padding mask
    #
    # No PAD tokens in this single sequence.
    # --------------------------------------------------

    padding_mask = torch.ones(
        1,
        x.size(1),
        dtype=torch.bool,
        device=device
    )

    # --------------------------------------------------
    # PREFILL
    #
    # Process entire prompt once.
    #
    # This creates the initial KV cache.
    # --------------------------------------------------

    (
        logits,
        _,
        past_key_values
    ) = model(
        x,
        padding_mask=padding_mask,
        use_cache=True
    )

    # Last token prediction
    next_logits = logits[:, -1, :]

    generated_ids = ids.copy()

    # --------------------------------------------------
    # DECODE
    #
    # From here onward we process ONLY one token.
    # --------------------------------------------------

    for _ in range(max_new_tokens):

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

        token_id = next_id.item()

        generated_ids.append(
            token_id
        )

        # --------------------------------------------------
        # Stop at EOS
        # --------------------------------------------------

        if token_id == EOS_ID:
            break

        # --------------------------------------------------
        # The new token is always a valid token.
        #
        # Add True to padding mask.
        # --------------------------------------------------

        padding_mask = torch.cat(
            [
                padding_mask,
                torch.ones(
                    1,
                    1,
                    dtype=torch.bool,
                    device=device
                )
            ],
            dim=1
        )

        # --------------------------------------------------
        # IMPORTANT:
        #
        # Only send the NEW token.
        #
        # NOT the entire sequence.
        # --------------------------------------------------

        (
            logits,
            _,
            past_key_values
        ) = model(
            next_id,
            padding_mask=padding_mask,
            past_key_values=past_key_values,
            use_cache=True
        )

        # Only one token was processed,
        # therefore logits shape is:
        #
        # [B, 1, vocab_size]
        #
        # We take the last token.
        next_logits = logits[:, -1, :]

    return [
        id_to_word[i]
        for i in generated_ids
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



"""
Comparison of token processing during training and inference:

+---------------------+-----------------------------------------------+----------------------------------------------+
| Feature             | Training                                      | Inference                                    |
+---------------------+-----------------------------------------------+----------------------------------------------+
| Token Processing    | Parallel: all tokens are processed at once.   | Sequential: autoregressive, one token at a   |
|                     |                                               | time.                                        |
+---------------------+-----------------------------------------------+----------------------------------------------+
| Attention Masking   | A causal mask is applied to a single large    | Past data is frozen; only the new token's    |
|                     | attention matrix.                             | Query is generated.                          |
+---------------------+-----------------------------------------------+----------------------------------------------+
| Weights (W_k, W_v)  | Dynamic: updated at every step via            | Static: locked in and frozen.                |
|                     | gradients.                                    |                                              |
+---------------------+-----------------------------------------------+----------------------------------------------+
| Computational Need  | Backpropagation requires storing all          | Only a forward pass for the newest token is  |
|                     | activations.                                  | required.                                    |
+---------------------+-----------------------------------------------+----------------------------------------------+
| KV Cache Status     | Not used and generally impossible to cache.   | Highly beneficial.                           |
+---------------------+-----------------------------------------------+----------------------------------------------+



During training
                     X_data
                       │
                       ▼
              ┌─────────────────┐
              │ Padding Mask     │
              │                 │
              │ T T T T F F ... │
              └────────┬────────┘
                       │
                       ▼
                Transformer
                       │
             ┌─────────┴─────────┐
             │                   │
        Causal Mask        Padding Mask
             │                   │
             └─────────┬─────────┘
                       │
                       ▼
                   Attention
                       │
                       ▼
                     FFN
                       │
                       ▼
                  Next Token
                  


During inference
                PROMPT
                  │
                  ▼
               PREFILL
                  │
                  ├──────────────┐
                  │              │
                  ▼              ▼
             KV CACHE       padding_mask
                  │              │
                  └──────┬───────┘
                         ▼
                      DECODE
                         │
                   one new token
                         │
                         ▼
                    update KV
                         │
                         ▼
                    next token


Now the function returns:
    output
    new K cache
    new V cache
    
What happens during generation?
Suppose:

Step 1:
    token = The

We calculate:
    K₁
    V₁

Cache:
K cache = [K₁]
V cache = [V₁]

Next:
Step 2:
    token = cat

Calculate only:
    Q₂
    K₂
    V₂

Then:
    K cache = [K₁, K₂]
    V cache = [V₁, V₂]

Next:
Step 3:
    token = sat

Calculate:
    Q₃
    K₃
    V₃

Cache becomes:
    K cache = [K₁, K₂, K₃]
    V cache = [V₁, V₂, V₃]

Imagine generating 1,000 tokens.

Without KV caching:
    step 1 → process 1 token
    step 2 → process 2 tokens
    step 3 → process 3 tokens
    ...
    step 1000 → process 1000 tokens

A huge amount of repeated computation occurs.
With KV cache:
    step 1 → new token
    step 2 → new token + cached K/V
    step 3 → new token + cached K/V
    ...
    step 1000 → new token + cached K/V

This makes autoregressive inference dramatically more efficient.
The tradeoff is memory: you now store K and V for every layer and every token in the context.

Suppose:
    batch = 1
    num_heads = 4
    sequence = 100
    head_dim = 16

Then:
K cache:
    [1, 4, 100, 16]

V cache:
    [1, 4, 100, 16]

For 12 Transformer layers:
    12 × K cache
    12 × V cache

So the KV cache can become a significant part of inference memory.

MHA → MQA → GQA
    Standard Multi-Head Attention:
        4 Q heads
        4 K heads
        4 V heads

    Q1 Q2 Q3 Q4
    K1 K2 K3 K4
    V1 V2 V3 V4

But KV cache memory is dominated by storing all those K/V heads.
So researchers introduced variants.

Multi-Query Attention (MQA)
    Q1 Q2 Q3 Q4
          │
         K
         V
    
    Many query heads share one K/V head.

Grouped-Query Attention (GQA)
    A compromise:
    
    Q1 Q2    Q3 Q4
     │ │      │ │
     K1 V1    K2 V2
    
    So:
        4 Q heads
        2 KV heads

GQA reduces KV-cache memory while retaining more expressive capacity than MQA.
Modern LLM architectures frequently use GQA or related approaches.

"""


