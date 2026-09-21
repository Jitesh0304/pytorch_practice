"""
Our current attention has:
    num_heads = 4

so:
    Q heads = 4
    K heads = 4
    V heads = 4

This is standard Multi-Head Attention (MHA).

The KV cache therefore stores:
    K1 K2 K3 K4
    V1 V2 V3 V4

    for every layer and every generated token. That consumes a lot of memory.

GQA changes this.

For example:
    4 Query heads
    2 Key heads
    2 Value heads

    Q1 ─┐
    Q2 ─┴── K1,V1

    Q3 ─┐
    Q4 ─┴── K2,V2

So multiple query heads share K/V heads.
This gives us:
    MHA
        4 Q + 4 KV

    GQA
        4 Q + 2 KV

    MQA
        4 Q + 1 KV

    The important consequence is:
        GQA dramatically reduces KV-cache size while retaining multiple query heads.

# -------------------------------------------
# Start with standard MHA
# -------------------------------------------

Suppose:
    d_model = 64
    num_query_heads = 8

    head_dim = 64/ 8 = 8

MHA has:
    Q heads = 8
    K heads = 8
    V heads = 8

Visually:
    Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8
    │  │  │  │  │  │  │  │
    K1 K2 K3 K4 K5 K6 K7 K8
    │  │  │  │  │  │  │  │
    V1 V2 V3 V4 V5 V6 V7 V8

Every query head gets its own K/V head.

for every:
    token × head

Suppose:
    sequence = 4096
    heads = 32
    head_dim = 128

Then each layer stores:
    K shape:
        [batch, 32, 4096, 128]

    V shape:
        [batch, 32, 4096, 128]

    That's a lot of memory. And during generation, this cache keeps growing.

# -------------------------------------------
# Multi-Query Attention
# -------------------------------------------
Researchers asked:
    Do we really need a separate K/V head for every query head?
    Maybe not.

MQA says:
    Q heads = 8
    K heads = 1
    V heads = 1

So:
    Q1 ─┐
    Q2 ─┤
    Q3 ─┤
    Q4 ─┤
    Q5 ─┤── K
    Q6 ─┤
    Q7 ─┤
    Q8 ─┘
         │
         V

    All query heads share the same K/V.
    This dramatically reduces KV-cache memory.

But MQA has a downside
Sharing a single K/V head can reduce the expressive capacity of attention.

So we want something between:
    MHA
and:
    MQA

That's where GQA comes in.

# -------------------------------------------
# Grouped Query Attention
# -------------------------------------------

Suppose:
    num_query_heads = 8
    num_kv_heads = 2

We divide the 8 query heads into 2 groups:

Group 1:
    Q1
    Q2
    Q3
    Q4
     │
     ├── K1
     └── V1

Group 2:
    Q5
    Q6
    Q7
    Q8
     │
     ├── K2
     └── V2

So:
    Q heads = 8
    KV heads = 2

Each KV head is shared by:
    8 / 2 = 4  query heads.

The key ratio
    Define:
        group_size = num_query_heads / num_KV_heads

    Example:
        group_size = 8/2 = 4
So:
    Q1 Q2 Q3 Q4 → KV1
    Q5 Q6 Q7 Q8 → KV2

Suppose:
    batch = 2
    sequence = 10
    d_model = 64
    query_heads = 8
    kv_heads = 2
    head_dim = 8

After projection:
    Q:       [2, 8, 10, 8]
    K:       [2, 2, 10, 8]
    V:       [2, 2, 10, 8]

Notice:
    Q → 8 heads
    K → 2 heads
    V → 2 heads

How can Q interact with K ?
    We need:
        Q:      [2, 8, 10, 8]
        K:      [2, 2, 10, 8]

    But attention expects matching head dimensions.
    So we repeat each K/V head:

    K = repeat_interleave(
        K,
        repeats=4,
        dim=1
    )

    Now:
        K:      [2, 8, 10, 8]

    Conceptually:
        Before:
            K1 K2

        After:
            K1 K1 K1 K1
            K2 K2 K2 K2

        Similarly:
            V = repeat_interleave(
                V,
                repeats=4,
                dim=1
            )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class GroupedQueryAttention(nn.Module):

    def __init__(
        self,
        d_model,
        num_query_heads,
        num_kv_heads,
        max_seq_len
    ):
        super().__init__()

        assert (
            num_query_heads % num_kv_heads == 0
        )

        self.d_model = d_model

        self.num_query_heads = (
            num_query_heads
        )

        self.num_kv_heads = (
            num_kv_heads
        )

        self.head_dim = (
            d_model // num_query_heads
        )

        self.group_size = (
            num_query_heads // num_kv_heads
        )

        self.W_Q = nn.Linear(
            d_model,
            num_query_heads * self.head_dim,
            bias=False
        )

        self.W_K = nn.Linear(
            d_model,
            num_kv_heads * self.head_dim,
            bias=False
        )

        self.W_V = nn.Linear(
            d_model,
            num_kv_heads * self.head_dim,
            bias=False
        )

        self.W_O = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.rope = RotaryEmbedding(
            self.head_dim,
            max_seq_len
        )

    def forward(
            self,
            x,
            padding_mask=None
    ):
        batch_size, seq_len, _ = x.shape

        Q = self.W_Q(x)

        K = self.W_K(x)

        V = self.W_V(x)

        Q = Q.view(
            batch_size,
            seq_len,
            self.num_query_heads,
            self.head_dim
        )

        Q = Q.transpose(1, 2)           # new shape [batch, 8, seq, 8]

        K = K.view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim
        )

        K = K.transpose(1, 2)               # new shape [batch, 2, seq, 8]

        V = V.view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim
        )

        V = V.transpose(1, 2)               # new shape [batch, 2, seq, 8]

        Q, K = self.rope(           # Q = [B, 8, S, 8]
            Q,
            K
        )

        K = K.repeat_interleave(    # K = [B, 8, S, 8]
            self.group_size,
            dim=1
        )

        V = V.repeat_interleave(    # V = [B, 8, S, 8]
            self.group_size,
            dim=1
        )

        scores = Q @ K.transpose(
            -2,
            -1
        )

        scores = scores / (
                self.head_dim ** 0.5
        )

        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            causal_mask == 0,
            float("-inf")
        )

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

        weights = F.softmax(
            scores,
            dim=-1
        )

        output = weights @ V

        output = output.transpose(
            1,
            2
        )

        output = output.contiguous().view(
            batch_size,
            seq_len,
            self.d_model
        )

        output = self.W_O(
            output
        )

        return output, weights, K, V


"""
So our complete attention mechanism is now:

                 Input
                   │
          ┌────────┼────────┐
          │        │        │
          ▼        ▼        ▼
          Q        K        V
          │        │        │
          │       RoPE      │
          │        │        │
          │        ▼        │
          │    2 KV heads   │
          │        │        │
          │    repeat ×4    │
          │        │        │
          └───────┬┴────────┘
                  ▼
              Attention
                  │
                  ▼
              W_O
              
              
GQA + KV Cache

Suppose:
    num_query_heads = 32
    num_kv_heads = 8

Then:
    group size = 4

During generation:
    New token
        │
        ▼
    Q → 32 heads
    K → 8 heads
    V → 8 heads

Cache:
    K cache → 8 heads
    V cache → 8 heads

We do not need:
    32 K heads
    32 V heads


Our current architecture is now:

                    Token IDs
                       │
                       ▼
                 Token Embedding
                       │
                       ▼
                    RMSNorm
                       │
                       ▼
              ┌─────────────────┐
              │      Q K V      │
              └────────┬────────┘
                       │
                 Q/K → RoPE
                       │
                       ▼
                      GQA
                       │
                 ┌─────┴─────┐
                 │           │
                 ▼           ▼
             KV Cache    Attention
                 │           │
                 └─────┬─────┘
                       ▼
                    Residual
                       │
                       ▼
                    RMSNorm
                       │
                       ▼
                    SwiGLU
                       │
                       ▼
                    Residual

# ------------------------------------------
# MHA vs GQA vs MQA
# ------------------------------------------

Architecture	Q Heads	    K Heads	    V Heads
    MHA	            8	        8	        8
    GQA	            8	        2	        2
    MQA	            8	        1	        1

The query side remains expressive, while the K/V side becomes cheaper.

The tradeoff is:
    MHA
    ↑ quality/capacity
    ↓ inference memory efficiency
    
    MQA
    ↑ memory efficiency
    ↓ potentially less expressive
    
    GQA
    middle ground
"""

class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_dim,
        max_seq_len,
        base=10000
    ):
        super().__init__()

        inv_freq = 1.0 / (
            base ** (
                torch.arange(
                    0,
                    head_dim,
                    2
                ).float()
                / head_dim
            )
        )

        positions = torch.arange(
            max_seq_len
        )

        freqs = torch.outer(
            positions,
            inv_freq
        )

        self.register_buffer(
            "cos",
            torch.cos(freqs)
        )

        self.register_buffer(
            "sin",
            torch.sin(freqs)
        )

    def forward(
        self,
        q,
        k,
        position_offset=0
    ):

        seq_len = q.shape[-2]

        cos = self.cos[
            position_offset:
            position_offset + seq_len
        ]

        sin = self.sin[
            position_offset:
            position_offset + seq_len
        ]

        q = self.rotate(
            q,
            cos,
            sin
        )

        k = self.rotate(
            k,
            cos,
            sin
        )

        return q, k

    def rotate(
        self,
        x,
        cos,
        sin
    ):

        x_even = x[..., 0::2]

        x_odd = x[..., 1::2]

        rotated_even = (
            x_even * cos
            -
            x_odd * sin
        )

        rotated_odd = (
            x_even * sin
            +
            x_odd * cos
        )

        x = torch.stack(
            [
                rotated_even,
                rotated_odd
            ],
            dim=-1
        )

        return x.flatten(-2)




class GQAAttention(nn.Module):

    def __init__(
        self,
        d_model,
        num_query_heads,
        num_kv_heads,
        max_seq_len
    ):
        super().__init__()

        assert (
            num_query_heads % num_kv_heads == 0
        )

        assert (
            d_model % num_query_heads == 0
        )

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads

        self.head_dim = (
            d_model // num_query_heads
        )

        self.group_size = (
            num_query_heads // num_kv_heads
        )

        self.q_proj = nn.Linear(
            d_model,
            num_query_heads * self.head_dim,
            bias=False
        )

        self.k_proj = nn.Linear(
            d_model,
            num_kv_heads * self.head_dim,
            bias=False
        )

        self.v_proj = nn.Linear(
            d_model,
            num_kv_heads * self.head_dim,
            bias=False
        )

        self.o_proj = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.rope = RotaryEmbedding(
            self.head_dim,
            max_seq_len
        )

    def forward(
        self,
        x,
        past_k=None,
        past_v=None
    ):

        B, T, _ = x.shape

        # -----------------------
        # Q K V projections
        # -----------------------

        Q = self.q_proj(x)

        K = self.k_proj(x)

        V = self.v_proj(x)

        # -----------------------
        # Reshape
        # -----------------------

        Q = Q.view(
            B,
            T,
            self.num_query_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            B,
            T,
            self.num_kv_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            B,
            T,
            self.num_kv_heads,
            self.head_dim
        ).transpose(1, 2)

        """
        Shapes:
            Q = [B, Q_heads, T, head_dim]
            K = [B, KV_heads, T, head_dim]
            V = [B, KV_heads, T, head_dim]
        """

        # -----------------------
        # Position offset
        # -----------------------

        if past_k is not None:
            position_offset = past_k.shape[2]               # past_k.shape = [B, 2, 5, 8]
        else:
            position_offset = 0

        # -----------------------
        # RoPE
        # -----------------------

        Q, K = self.rope(
            Q,
            K,
            position_offset
        )

        # -----------------------
        # Cache
        # -----------------------

        if past_k is not None:

            K = torch.cat(
                [past_k, K],
                dim=2
            )

        if past_v is not None:

            V = torch.cat(
                [past_v, V],
                dim=2
            )

        """
        For example:
            past K:
                [B, 2, 5, 8]
            new K:
                [B, 2, 1, 8]
            After concatenation:
                [B, 2, 6, 8]
        """

        # -----------------------
        # GQA
        # -----------------------

        K_attn = K.repeat_interleave(
            self.group_size,
            dim=1
        )

        V_attn = V.repeat_interleave(
            self.group_size,
            dim=1
        )

        """
        So:
            Q:
                [B, 8, T, 8]
            K:
                [B, 8, total_T, 8]
            V:
                [B, 8, total_T, 8]
        """

        # -----------------------
        # Attention
        # -----------------------

        scores = (
            Q @ K_attn.transpose(-2, -1)
        )

        scores = scores / (
            self.head_dim ** 0.5
        )

        # -----------------------
        # Causal mask
        # -----------------------

        if past_k is None:

            causal_mask = torch.tril(
                torch.ones(
                    T,
                    T,
                    device=x.device
                )
            )

            scores = scores.masked_fill(
                causal_mask == 0,
                float("-inf")
            )

        # -----------------------
        # Softmax
        # -----------------------
        weights = F.softmax(
            scores,
            dim=-1
        )

        # -----------------------
        # Weighted values
        # -----------------------

        output = weights @ V_attn

        # -----------------------
        # Merge heads
        # -----------------------

        output = output.transpose(
            1,
            2
        )

        output = output.contiguous().view(
            B,
            T,
            self.d_model
        )

        output = self.o_proj(
            output
        )

        return output, weights, K, V


d_model = 32
query_heads = 4
kv_heads = 2
max_seq_len = 20

attention = GQAAttention(
    d_model,
    query_heads,
    kv_heads,
    max_seq_len
)

x = torch.randn(
    1,
    5,
    32
)

output, weights, K, V = attention(x)

print(output.shape)
print(K.shape)
print(V.shape)


"""
Visualize generation
    Imagine:

Step 1
    "The"

Cache:
K = [The]
V = [The]

Step 2
    "cat"

Cache:
K = [The, cat]
V = [The, cat]

Step 3
    "sat"

Cache:
K = [The, cat, sat]
V = [The, cat, sat]

Step 4
    "on"

Cache:
K = [The, cat, sat, on]
V = [The, cat, sat, on]

Step 5
    "the"

Cache:
K = [The, cat, sat, on, the]
V = [The, cat, sat, on, the]

The important thing:
We don't recalculate K/V for the old tokens.

Our Transformer block is now:

                       Input
                         │
                         ▼
                      RMSNorm
                         │
                         ▼
                    GQA Attention
                         │
                  ┌──────┴──────┐
                  │             │
                  Q             K
                  │             │
                RoPE          RoPE
                  │             │
                  └──────┬──────┘
                         │
                       QKᵀ
                         │
                      Softmax
                         │
                         V
                         │
                    KV Cache
                         │
                         ▼
                      Output
                         │
                         ▼
                     Residual
                         │
                         ▼
                      RMSNorm
                         │
                         ▼
                      SwiGLU
                         │
                         ▼
                      Residual

# -----------------
# Training 
# -----------------

We normally process the whole sequence:
    "The cat sat on the mat"
all at once.

Q → entire sequence
K → entire sequence
V → entire sequence

No KV cache is normally necessary.
The causal mask prevents looking into the future.

# -----------------
# Inference
# -----------------

We generate:

The
↓
cat
↓
sat
↓
on
↓
the
↓
mat

one token at a time.
Now KV cache becomes extremely useful.
Previous tokens
      │
      ▼
   KV Cache
      │
      │
New token → Q
New token → K,V → append
"""


