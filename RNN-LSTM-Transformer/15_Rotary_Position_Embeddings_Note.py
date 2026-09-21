"""
RoPE (Rotary Positional Embeddings) vs Positional Encoding
============================================================

WHY DO MODERN TRANSFORMERS USE RoPE?

The main reason modern Transformer/LLM architectures use RoPE instead of
traditional absolute positional encoding is that RoPE injects positional
information directly into the Query (Q) and Key (K) representations used
by attention.

This gives the attention mechanism a natural way to represent RELATIVE
POSITION between tokens.

---------------------------------------------------------------------------
1. ORIGINAL TRANSFORMER POSITIONAL ENCODING
---------------------------------------------------------------------------

In the original Transformer, positional information is added directly to
the token embedding:

    X = TokenEmbedding + PositionalEncoding

For example:

    Tokens:       I       love       AI
    Position:     0        1         2

    Embedding:   E0       E1        E2
    Position:    P0       P1        P2

    Input:
                E0 + P0   E1 + P1   E2 + P2


The Transformer then computes:

    Q = (X + P) W_Q
    K = (X + P) W_K
    V = (X + P) W_V

Attention is:

    Attention(Q, K, V)
        = softmax(Q K^T / sqrt(d_k)) V


The original Transformer uses sinusoidal positional encoding:

    PE(pos, 2i)
        = sin(pos / 10000^(2i / d))

    PE(pos, 2i + 1)
        = cos(pos / 10000^(2i / d))


The important point is:

    POSITION INFORMATION IS ADDED TO THE INPUT EMBEDDING.

The model then has to learn how this positional information should affect
the Q/K attention interaction.

---------------------------------------------------------------------------
2. WHAT IS DIFFERENT ABOUT RoPE?
---------------------------------------------------------------------------

RoPE does NOT simply add a positional vector to the token embedding.

Instead, RoPE rotates the Query and Key vectors according to their positions.

Conceptually:

    Original Q
        |
        | rotate according to position m
        v
    Rotated Q

    Original K
        |
        | rotate according to position n
        v
    Rotated K


Mathematically:

    Q' = R_m Q
    K' = R_n K

where:

    m = position of the Query token
    n = position of the Key token
    R_m = rotation matrix corresponding to position m
    R_n = rotation matrix corresponding to position n


Attention then uses:

    Q' K'^T

Instead of adding position information to X, RoPE modifies Q and K
before calculating attention.

---------------------------------------------------------------------------
3. THE MOST IMPORTANT PROPERTY OF RoPE
---------------------------------------------------------------------------

Consider:

    Q' = R_m Q
    K' = R_n K

The attention score is:

    Q'^T K'

Substituting the rotated representations:

    (R_m Q)^T (R_n K)

Using transpose properties:

    = Q^T R_m^T R_n K

For rotation matrices:

    R_m^T = R_-m

Therefore:

    R_m^T R_n
        = R_-m R_n
        = R_(n-m)

So:

    (R_m Q)^T (R_n K)
        = Q^T R_(n-m) K


The important result is:

    POSITION INFORMATION DEPENDS ON (n - m)

That means the attention interaction naturally contains the RELATIVE
DISTANCE between the Query and Key positions.


For example:

    Query position = 10
    Key position   = 6

    Relative distance:

        6 - 10 = -4


RoPE can represent this positional relationship through the rotation:

    R_-4


This is one of the most important mathematical reasons RoPE is useful.

---------------------------------------------------------------------------
4. ABSOLUTE POSITION VS RELATIVE POSITION
---------------------------------------------------------------------------

Traditional absolute positional encoding tells the model something like:

    "This token is at position 37."

RoPE gives the attention mechanism information that allows it to represent:

    "This token is 5 positions away from me."


This distinction is important because many linguistic relationships are
more naturally described using relative distances.

For example:

    The cat sat on the mat.

Positions:

    The   = 0
    cat   = 1
    sat   = 2
    on    = 3
    the   = 4
    mat   = 5


Distance between:

    cat and mat:

        5 - 1 = 4


Another sentence could have a similar relationship:

    The dog was sitting on the floor.

If:

    dog   = 1
    floor = 5

then:

    5 - 1 = 4


The absolute positions are not necessarily important.

The relative relationship can be more useful.

---------------------------------------------------------------------------
5. WHY IS RoPE APPLIED TO Q AND K?
---------------------------------------------------------------------------

Attention fundamentally depends on the similarity between Q and K:

    Attention score = Q K^T


Therefore, if we want positional information to influence how tokens
interact with each other, modifying Q and K is a very direct approach.

With absolute positional encoding:

    Token embedding
          +
    Position embedding
          |
          v
        Input
          |
          v
        Q / K
          |
          v
      Attention


With RoPE:

    Token embedding
          |
          v
        Q / K
          |
          v
    Position-dependent rotation
          |
          v
     Rotated Q / K
          |
          v
      Attention


Therefore, RoPE injects positional information directly into the
attention calculation.

---------------------------------------------------------------------------
6. RoPE DOES NOT CHANGE THE MAGNITUDE OF THE VECTOR
---------------------------------------------------------------------------

A rotation preserves vector magnitude.

For a 2D vector:

    x = [x0, x1]

RoPE applies:

        [ cos(theta)  -sin(theta) ]
    R = [ sin(theta)   cos(theta) ]

Then:

    x' = R x


Therefore:

    x0' = x0 cos(theta) - x1 sin(theta)

    x1' = x0 sin(theta) + x1 cos(theta)


Because this is a rotation:

    ||x'|| = ||x||


So RoPE changes the ORIENTATION of the vector according to its position,
rather than simply adding another vector to it.

---------------------------------------------------------------------------
7. HOW DOES RoPE ROTATE A HIGH-DIMENSIONAL VECTOR?
---------------------------------------------------------------------------

Suppose:

    Q = [q0, q1, q2, q3, q4, q5, ...]


RoPE treats dimensions in pairs:

    [q0, q1]  -> rotate
    [q2, q3]  -> rotate
    [q4, q5]  -> rotate
    ...


Each pair receives a position-dependent rotation.

Conceptually:

    [q0, q1] -> R(theta_0)
    [q2, q3] -> R(theta_1)
    [q4, q5] -> R(theta_2)
    ...


Different dimension pairs use different frequencies.

This allows the representation to encode positional information at
different scales.

---------------------------------------------------------------------------
8. POSITION-DEPENDENT ROTATION
---------------------------------------------------------------------------

For position m, the rotation angle can be represented as:

    theta = m * omega

where:

    m     = token position
    omega = frequency associated with the dimension pair


Different dimensions use different frequencies.

Therefore:

    Position 0 -> rotation based on 0
    Position 1 -> rotation based on 1
    Position 2 -> rotation based on 2
    ...
    Position m -> rotation based on m


This creates a deterministic positional transformation.

There is no need to learn a separate embedding vector for every position.

---------------------------------------------------------------------------
9. LEARNED ABSOLUTE POSITION EMBEDDINGS HAVE A LIMIT
---------------------------------------------------------------------------

A learned absolute positional embedding is commonly implemented as:

    position_embedding = nn.Embedding(max_seq_len, d_model)


For example:

    max_seq_len = 2048


The model learns:

    P0
    P1
    P2
    ...
    P2047


What happens at:

    position 2048?
    position 2049?
    position 3000?


Those positions do not have learned embedding vectors.

Therefore, learned absolute positional embeddings have a natural
maximum sequence length unless additional mechanisms are introduced.

---------------------------------------------------------------------------
10. RoPE AND LONGER CONTEXT
---------------------------------------------------------------------------

RoPE does not require a separate learned embedding vector for every
position.

Instead, it generates a rotation mathematically from the position.

Therefore, it is naturally more suitable for extending context than a
simple learned absolute position embedding table.

However:

    RoPE is NOT infinitely extrapolatable.

Standard RoPE can still degrade when the model is used far beyond the
context length seen during training.

Because of this, modern LLMs use various RoPE scaling techniques for
long-context models.

Examples include:

    - RoPE scaling
    - NTK-aware scaling
    - YaRN
    - Other context-extension techniques


The important distinction is:

    Learned absolute PE:
        position must generally exist in the learned table.

    RoPE:
        position is generated mathematically.


---------------------------------------------------------------------------
11. RoPE AND AUTOREGRESSIVE LLMs
---------------------------------------------------------------------------

RoPE works particularly naturally with autoregressive language models.

Suppose the sequence is:

    I love machine learning


Positions:

    I          -> 0
    love       -> 1
    machine    -> 2
    learning   -> 3


When generating the next token:

    AI -> position 4


The new Query receives the rotation corresponding to position 4:

    Q_4' = R_4 Q_4


The cached Keys retain their positional transformations:

    K_0' = R_0 K_0
    K_1' = R_1 K_1
    K_2' = R_2 K_2
    K_3' = R_3 K_3


The attention interaction therefore naturally contains:

    R_4^T R_0 -> relative distance -4
    R_4^T R_1 -> relative distance -3
    R_4^T R_2 -> relative distance -2
    R_4^T R_3 -> relative distance -1


This works very well with KV caching.

---------------------------------------------------------------------------
12. RoPE AND KV CACHE
---------------------------------------------------------------------------

During autoregressive generation, we normally cache:

    K
    V


For previous tokens:

    K_0, K_1, K_2, ...
    V_0, V_1, V_2, ...


When a new token arrives:

    Q_new
    K_new
    V_new


We only need to calculate the new Q/K/V.

With RoPE:

    Q_new -> rotate according to current position
    K_new -> rotate according to current position


Then:

    Attention(Q_new, K_cached, V_cached)


The positional relationship between the current Query and cached Keys
is represented through the difference in their rotation angles.

This is one reason RoPE fits naturally into modern autoregressive LLMs.

---------------------------------------------------------------------------
13. RoPE VS RELATIVE POSITIONAL ENCODING
---------------------------------------------------------------------------

Another approach is explicit Relative Positional Encoding.

Instead of rotating Q/K, one can modify the attention score:

    Attention score
        =
    Q K^T + RelativePositionBias


Conceptually:

    distance     bias
    -----------------
       -2         ...
       -1         ...
        0         ...
       +1         ...
       +2         ...


This explicitly adds a positional bias to the attention scores.

RoPE takes another approach:

    Q -> R_m Q
    K -> R_n K

Then:

    Q'^T K'
        =
    Q^T R_(n-m) K


So relative position emerges naturally from the interaction between
the rotated Query and Key.

---------------------------------------------------------------------------
14. IMPORTANT COMPARISON
---------------------------------------------------------------------------

Absolute Positional Encoding:

    X' = X + P

    Position is added to the token representation.

RoPE:

    Q' = R_m Q
    K' = R_n K

    Position modifies Q and K through rotation.

The key conceptual difference:

    Absolute PE:
        "Where am I?"

    RoPE:
        "How are these two positions related?"


---------------------------------------------------------------------------
15. ABSOLUTE PE VS RoPE
---------------------------------------------------------------------------

Property                         Absolute PE       RoPE
---------------------------------------------------------------------------
Adds position to embedding      Yes                No
Rotates Q/K                     No                 Yes
Directly affects attention      Indirectly         Directly
Relative position               Less natural       Natural
Learned position table          Sometimes          No
KV-cache friendly               Yes                Yes
Long-context extension          More difficult     More natural
Used in modern LLMs              Less common        Very common
Mathematical simplicity          Moderate           Strong


---------------------------------------------------------------------------
16. THE CORE MATHEMATICAL IDEA
---------------------------------------------------------------------------

The most important equation to remember is:

    Q_m' = R_m Q

    K_n' = R_n K


Then:

    Q_m'^T K_n'

    = (R_m Q)^T (R_n K)

    = Q^T R_m^T R_n K

    = Q^T R_(n-m) K


Therefore:

    +---------------------------------------------+
    |                                             |
    |   RoPE attention depends on relative        |
    |   positional relationship (n - m).          |
    |                                             |
    +---------------------------------------------+


This is the core mathematical intuition behind RoPE.


---------------------------------------------------------------------------
17. WHY RESEARCHERS PREFER RoPE IN MANY MODERN LLMs
---------------------------------------------------------------------------

RoPE provides several useful properties:

1. Relative position emerges naturally.

2. Position is injected directly into Q/K attention.

3. It does not require a learned positional embedding table.

4. It works naturally with autoregressive decoding.

5. It works well with KV caching.

6. Rotation preserves vector magnitude.

7. Different dimensions can encode different positional frequencies.

8. It is more naturally adaptable to longer context than learned
   absolute positional embeddings.

9. The mathematical formulation is elegant and computationally efficient.


---------------------------------------------------------------------------
18. SIMPLE INTUITION
---------------------------------------------------------------------------

Think of absolute positional encoding as giving each token an address:

    Token A:
        "I am at position 10."

    Token B:
        "I am at position 15."


RoPE allows the attention mechanism to naturally reason about:

    "A and B are 5 positions apart."


So:

    Absolute PE
        |
        v
    Absolute location


    RoPE
        |
        v
    Relative positional relationship


This is especially useful because attention is fundamentally about
relationships between tokens.


---------------------------------------------------------------------------
19. ONE-SENTENCE INTERVIEW ANSWER
---------------------------------------------------------------------------

If asked in an interview:

    "Why is RoPE preferred over traditional positional encoding?"

A strong answer is:

    "RoPE injects positional information directly into the Query and Key
    representations through position-dependent rotations. Because the
    dot product between rotated Q and K depends on the relative position
    difference, RoPE gives attention a natural representation of relative
    positions while remaining efficient and compatible with KV caching.
    It also avoids a learned positional embedding table and is more
    naturally adaptable to longer contexts."


---------------------------------------------------------------------------
20. FINAL TAKEAWAY
---------------------------------------------------------------------------

Traditional absolute positional encoding:

    Token
      +
    Position
      |
      v
    Transformer


RoPE:

    Token
      |
      v
    Q / K
      |
      v
    Position-dependent rotation
      |
      v
    Attention


The key mathematical property:

    (R_m Q)^T (R_n K)
        =
    Q^T R_(n-m) K


Therefore:

    RoPE converts absolute token positions into a positional relationship
    that naturally appears inside the Q/K attention interaction.


The single most important thing to remember:

    RoPE does not simply tell the model where a token is.

    It changes Q and K so that their attention interaction naturally
    contains information about how far apart their positions are.


############################################################################
#################                Example                   #################
############################################################################

RoPE Example Using the Sentence:
    "I love machine learning"

====================================================================
1. TOKEN POSITIONS
====================================================================

Suppose our tokenizer produces:

    Token        Position
    ---------------------
    I                0
    love             1
    machine          2
    learning         3


So the sequence is:

    I -> love -> machine -> learning
    0      1        2          3


In a Transformer, every token eventually produces:

    Q (Query)
    K (Key)
    V (Value)


For understanding RoPE, we only need Q and K.

====================================================================
2. SUPPOSE WE HAVE SIMPLE Q AND K VECTORS
====================================================================

To make the example easy, assume every token has a 2-dimensional
Query and Key vector.

These values are completely artificial and are only for understanding.

    Token       Q                  K
    -----------------------------------------
    I           [1.0, 0.0]         [1.0, 0.0]
    love        [1.0, 0.0]         [1.0, 0.0]
    machine     [1.0, 0.0]         [1.0, 0.0]
    learning    [1.0, 0.0]         [1.0, 0.0]


Without RoPE, all of these vectors have exactly the same direction.

So if we calculate:

    Q @ K

for any pair, we get:

    [1, 0] @ [1, 0] = 1


The model therefore needs positional information to distinguish:

    I
    love
    machine
    learning


====================================================================
3. ROPE ASSIGNS A ROTATION BASED ON POSITION
====================================================================

For this simple example, let's use:

    angle = position * theta

and choose:

    theta = 45 degrees


This is intentionally simplified.

Real LLMs use different frequencies for different dimension pairs.


Therefore:

    Position       Rotation
    -----------------------
       0              0°
       1             45°
       2             90°
       3            135°


So our sentence becomes:

    I          -> position 0 -> rotate 0°
    love       -> position 1 -> rotate 45°
    machine    -> position 2 -> rotate 90°
    learning   -> position 3 -> rotate 135°


====================================================================
4. ROTATING "I"
====================================================================

"I" is at position 0.

Original:

    Q_I = [1, 0]

Rotation:

    0 degrees


Therefore:

    Q'_I = [1, 0]


Visually:

                  ↑
                  |
                  |
                  |
                  +-------->
                         Q_I


No rotation occurs.

====================================================================
5. ROTATING "LOVE"
====================================================================

"love" is at position 1.

Original:

    Q_love = [1, 0]

Rotation:

    45 degrees


The rotation equations are:

    x' = x*cos(theta) - y*sin(theta)

    y' = x*sin(theta) + y*cos(theta)


For:

    x = 1
    y = 0
    theta = 45°


We get approximately:

    x' = 0.707
    y' = 0.707


Therefore:

    Q'_love = [0.707, 0.707]


The vector has now rotated.

====================================================================
6. ROTATING "MACHINE"
====================================================================

"machine" is at position 2.

Rotation:

    2 * 45°
    = 90°


Original:

    Q_machine = [1, 0]


After rotation:

    Q'_machine = [0, 1]


So:

    machine
       |
       |  Q
       ↑


====================================================================
7. ROTATING "LEARNING"
====================================================================

"learning" is at position 3.

Rotation:

    3 * 45°
    = 135°


Original:

    Q_learning = [1, 0]


After rotation:

    Q'_learning ≈ [-0.707, 0.707]


So each token now has a different orientation:

    I          -> [ 1.000,  0.000]
    love       -> [ 0.707,  0.707]
    machine    -> [ 0.000,  1.000]
    learning   -> [-0.707,  0.707]


====================================================================
8. WHY IS THIS USEFUL?
====================================================================

Now consider attention between:

    "I"

and:

    "machine"


Their positions are:

    I       = 0
    machine = 2


Therefore:

    relative distance = 2 - 0
                      = 2


Their rotation difference is:

    90° - 0°
    = 90°


Their vectors are:

    Q'_I       = [1, 0]

    K'_machine = [0, 1]


The attention dot product is:

    Q'_I @ K'_machine

    = [1, 0] @ [0, 1]

    = 0


The positional relationship therefore affects the attention score.


====================================================================
9. ATTENTION BETWEEN "I" AND "LOVE"
====================================================================

Positions:

    I    = 0
    love = 1


Relative distance:

    1 - 0 = 1


Rotation difference:

    45°


Vectors:

    Q'_I    = [1, 0]

    K'_love = [0.707, 0.707]


Dot product:

    [1, 0] @ [0.707, 0.707]

    = 0.707


So:

    I -> love

gets a positional contribution corresponding to a 45° difference.


====================================================================
10. ATTENTION BETWEEN "I" AND "LEARNING"
====================================================================

Positions:

    I        = 0
    learning = 3


Relative distance:

    3 - 0 = 3


Rotation difference:

    135°


Vectors:

    Q'_I        = [1, 0]

    K'_learning = [-0.707, 0.707]


Dot product:

    [1, 0] @ [-0.707, 0.707]

    = -0.707


Again, the attention interaction is affected by the relative
position.


====================================================================
11. VERY IMPORTANT: THE SAME DISTANCE GIVES THE SAME RELATIONSHIP
====================================================================

Consider:

    I       -> machine

Positions:

    0 -> 2

Distance:

    2


Now consider:

    love -> learning

Positions:

    1 -> 3

Distance:

    2


Both have:

    relative distance = 2


Their rotation difference is also:

    90°


This is the important property of RoPE.


    I ---------> machine
    0               2
          distance 2


    love -------> learning
      1              3
          distance 2


Even though the absolute positions are different:

    (0, 2)

and:

    (1, 3)


their relative distance is the same:

    2


RoPE therefore produces the same relative rotation.


====================================================================
12. THIS IS THE MATHEMATICAL IDEA
====================================================================

For Query at position m:

    Q'_m = R_m Q


For Key at position n:

    K'_n = R_n K


Their attention interaction is:

    Q'_m^T K'_n

    = (R_m Q)^T (R_n K)

    = Q^T R_m^T R_n K

    = Q^T R_(n-m) K


The important part is:

    n - m


which is the relative distance between the two tokens.


====================================================================
13. APPLYING THIS TO OUR SENTENCE
====================================================================

Sentence:

    "I love machine learning"


Positions:

    I          = 0
    love       = 1
    machine    = 2
    learning   = 3


Suppose we ask:

    How does "learning" attend to previous tokens?


The Query is:

    Q_learning

at:

    position = 3


The Keys are:

    K_I
        position = 0

    K_love
        position = 1

    K_machine
        position = 2


The relative distances are:

    learning -> I

        0 - 3 = -3


    learning -> love

        1 - 3 = -2


    learning -> machine

        2 - 3 = -1


Therefore RoPE gives the attention mechanism positional relationships
corresponding to:

    -3
    -2
    -1


Conceptually:

                 learning
                    |
        +-----------+-----------+
        |           |           |
        ↓           ↓           ↓
       I           love       machine

       -3           -2          -1


The model can therefore distinguish:

    "machine is immediately before learning"

from:

    "I is three positions before learning."


====================================================================
14. IMPORTANT DISTINCTION
====================================================================

RoPE does NOT mean that:

    I
    love
    machine
    learning


will always have these exact Q/K vectors:

    [1,0]
    [0.707,0.707]
    [0,1]
    [-0.707,0.707]


Those vectors were invented for this example.

In a real Transformer:

    Token embedding
          ↓
    Transformer layers
          ↓
    Q and K projections
          ↓
    RoPE rotation
          ↓
    Attention


The Q/K vectors are high-dimensional learned representations.


====================================================================
15. REAL RoPE USES MANY DIMENSIONS
====================================================================

Suppose:

    head_dim = 8


The vector could be:

    Q = [
        q0, q1,
        q2, q3,
        q4, q5,
        q6, q7
    ]


RoPE pairs dimensions:

    (q0, q1)
    (q2, q3)
    (q4, q5)
    (q6, q7)


Each pair is rotated:

    (q0, q1) -> frequency 0
    (q2, q3) -> frequency 1
    (q4, q5) -> frequency 2
    (q6, q7) -> frequency 3


So a real Q vector is not simply rotated in one 2D plane.

It is effectively rotated across multiple 2D dimension pairs, with
different frequencies.


====================================================================
16. SIMPLE PYTHON VERSION
====================================================================

The following code demonstrates the basic idea:

    import math


    def rotate(vector, position, theta=math.pi / 4):

        angle = position * theta

        cos_theta = math.cos(angle)
        sin_theta = math.sin(angle)

        x, y = vector

        return [
            x * cos_theta - y * sin_theta,
            x * sin_theta + y * cos_theta,
        ]


    tokens = [
        "I",
        "love",
        "machine",
        "learning",
    ]


    q = [1.0, 0.0]


    for position, token in enumerate(tokens):

        rotated_q = rotate(q, position)

        print(
            token,
            position,
            rotated_q
        )


Expected approximate result:

    I          0    [1.000, 0.000]
    love       1    [0.707, 0.707]
    machine    2    [0.000, 1.000]
    learning   3    [-0.707, 0.707]


====================================================================
17. TESTING RELATIVE POSITION
====================================================================

Let's compare:

    I -> machine

and:

    love -> learning


Positions:

    I        = 0
    machine  = 2

    love     = 1
    learning = 3


Both have:

    distance = 2


We can verify their relative rotation:

    R_0^T R_2

and:

    R_1^T R_3


Both result in:

    R_2


Therefore RoPE represents the same relative positional relationship.


====================================================================
18. THE BIG PICTURE
====================================================================

Our sentence:

    I       love       machine       learning
    |         |           |              |
    0         1           2              3
    |         |           |              |
    ↓         ↓           ↓              ↓
   Q/K       Q/K         Q/K            Q/K
    |         |           |              |
    ↓         ↓           ↓              ↓
   R0        R1          R2             R3
    |         |           |              |
    ↓         ↓           ↓              ↓
 rotated   rotated     rotated        rotated
   Q/K       Q/K         Q/K            Q/K
      \        |          |             /
       \       |          |            /
        +------+----------+-----------+
                       |
                       ↓
                  Q K^T
                       |
                       ↓
                   Attention


The rotation encodes position into Q/K.

When Q and K interact, their relative rotations encode their
relative positions.


====================================================================
19. THE ONE THING TO REMEMBER
====================================================================

For:

    "I love machine learning"


the token positions are:

    I          -> 0
    love       -> 1
    machine    -> 2
    learning   -> 3


RoPE does approximately:

    Q_I        -> rotate by 0
    Q_love     -> rotate by position 1
    Q_machine  -> rotate by position 2
    Q_learning -> rotate by position 3


Then when attention calculates:

    Q_m^T K_n


the relative rotation becomes:

    R_(n-m)


Therefore:

    I -> machine

and:

    love -> learning


have the same relative distance:

    2


and therefore the same relative positional rotation.


====================================================================
FINAL INTUITION
====================================================================

Without RoPE:

    "I"       = position 0
    "love"    = position 1
    "machine" = position 2
    "learning"= position 3


Position is added as separate information.


With RoPE:

    "I"       -> vector rotated by position 0
    "love"    -> vector rotated by position 1
    "machine" -> vector rotated by position 2
    "learning"-> vector rotated by position 3


Then:

    Query rotation
          +
    Key rotation
          ↓
    relative rotation
          ↓
    relative position
          ↓
    attention score


The central equation is:

    (R_m Q)^T (R_n K)
        =
    Q^T R_(n-m) K


So for the sentence:

    "I love machine learning"


RoPE allows attention to naturally encode relationships such as:

    learning is 1 position after machine
    learning is 2 positions after love
    learning is 3 positions after I

rather than only representing each token as having an independent
absolute position.

"""

"""
With RoPE, we don't add a position vector.
Instead:
    Token
      ↓
    Token Embedding
      ↓
    Q K V
      ↓
    Rotate Q and K
      ↓
    Attention

So the architecture becomes:
                 Token IDs
                    │
                    ▼
              Token Embedding
                    │
                    ▼
              Transformer Block
                    │
             ┌──────┴──────┐
             │             │
             ▼             ▼
             Q             K
             │             │
             ▼             ▼
           RoPE           RoPE
             │             │
             └──────┬──────┘
                    ▼
               Q @ Kᵀ
                    │
                    ▼
                Attention

Notice:
    RoPE is applied to Q and K, not to the input embeddings.

We don't need to modify V.
Therefore:
    Q → RoPE
    K → RoPE
    V → unchanged

Why doesn't RoPE rotate V?
    Because position is primarily needed to determine:

Which tokens should attend to which other tokens?
    That relationship is determined through:
        QK^T

So we inject position into:
    Q,K

while leaving V unchanged.

A useful mental model is:
    Q = "what am I looking for?"
    K = "what information do I represent?"
    V = "what information should I provide?"

RoPE modifies the positional relationship between the query and key.

The really beautiful part
Consider two tokens:
    token A → position 2
    token B → position 5

After rotation, their dot product contains information related to the difference:
    5 - 2 = 3
So attention can learn relationships based on relative position.
This is one of the elegant mathematical properties of RoPE.

                    Mini GPT
                       │
                       ▼
                Token Embedding
                       │
                       ▼
              Transformer Block
                       │
             ┌─────────┴─────────┐
             │                   │
          LayerNorm          Attention
                                 │
                           ┌─────┴─────┐
                           │           │
                           Q           K
                           │           │
                         RoPE        RoPE
                           │           │
                           └─────┬─────┘
                                 │
                              QKᵀ
                                 │
                              Softmax
                                 │
                                 V
                                 │
                           Attention
                                 │
                            Residual
                                 │
                              LayerNorm
                                 │
                                FFN
                                 │
                              Residual
                                 │
                                ...

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def rotate(
            self,
            x
    ):
        seq_len = x.shape[-2]

        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rotated_even = (
                x_even * cos
                -
                x_odd * sin
        )

        x_rotated_odd = (
                x_even * sin
                +
                x_odd * cos
        )
        x_rotated = torch.stack(
            [
                x_rotated_even,
                x_rotated_odd
            ],
            dim=-1
        )

        return x_rotated.flatten(
            -2
        )

    def forward(
            self,
            q,
            k
    ):
        q = self.rotate(q)

        k = self.rotate(k)

        return q, k


# rope = RotaryEmbedding(
#     head_dim=8,
#     max_seq_len=10
# )
#
# q = torch.randn(            # [batch, heads, sequence, head_dim]
#     1,
#     4,
#     5,
#     8
# )
#
# k = torch.randn(
#     1,
#     4,
#     5,
#     8
# )
#
#
# q_rot, k_rot = rope(
#     q,
#     k
# )
#
# print(q_rot.shape)
# print(k_rot.shape)


class MultiHeadSelfAttention(nn.Module):

    def __init__(
            self,
            d_model,
            num_heads,
            max_seq_len
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = (
                d_model // num_heads
        )

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

        self.rope = RotaryEmbedding(
            self.head_dim,
            max_seq_len
        )

    def forward(
            self,
            x,
            padding_mask=None,
            past_k=None,
            past_v=None
    ):

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
        # Apply RoPE
        # -------------------------

        Q, K = self.rope(
            Q,
            K
        )

        # -------------------------
        # KV cache
        # -------------------------

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

        # -------------------------
        # Attention
        # -------------------------

        scores = Q @ K.transpose(
            -2,
            -1
        )

        scores = scores / (
                self.head_dim ** 0.5
        )

        # -------------------------
        # Causal mask
        # -------------------------

        total_len = K.shape[-2]

        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                total_len,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            causal_mask == 0,
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

        weights = F.softmax(
            scores,
            dim=-1
        )

        # -------------------------
        # Weighted values
        # -------------------------

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

        output = self.W_O(output)

        return output, weights, K, V


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


"""
One correction before we move on

Our current code has two advanced features:
    RoPE
    +
    KV cache

but they're not yet fully integrated for incremental generation.
For correct cached generation, we need RoPE to know:
    "Where in the sequence is this new token?"

For example:

    Initial prompt:
        The cat sat

    positions:
        0 1 2

    Then we generate:
        on

    Its position must be:
        3        not   0

    So the final attention API should eventually receive something like:
        position_offset = past_k.shape[-2]

    and calculate RoPE using:
        position = offset + local_position
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


"""
Now we can say:

rope(
    q,
    k,
    position_offset=5
)

and the token gets position:
    5

instead of:
    0

"""