import math
import torch
import torch.nn as nn
import torch.nn.functional as F


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

corpus = [
    "low lower lowest",
    "low lower",
    "low lowest",
    "new newer newest"
]

words = [
    "low",
    "lower",
    "lowest",
    "new",
    "newer",
    "newest"
]

def get_initial_vocab(words):

    vocab = set()

    for word in words:

        for char in word:

            vocab.add(char)

    return vocab

vocab = get_initial_vocab(words)
print(vocab)

word_tokens = {
    "low": ["l", "o", "w"],
    "lower": ["l", "o", "w", "e", "r"],
    "lowest": ["l", "o", "w", "e", "s", "t"],
    "new": ["n", "e", "w"],
    "newer": ["n", "e", "w", "e", "r"],
    "newest": ["n", "e", "w", "e", "s", "t"]
}

def initialize_words(words):

    word_tokens = {}

    for word in words:

        word_tokens[word] = list(word)

    return word_tokens

word_tokens = initialize_words(words)

print(word_tokens["lower"])


from collections import Counter


def count_pairs(
    word_tokens
):

    pair_counts = Counter()

    for tokens in word_tokens.values():

        for i in range(
            len(tokens) - 1
        ):

            pair = (
                tokens[i],
                tokens[i + 1]
            )

            pair_counts[pair] += 1

    return pair_counts


pair_counts = count_pairs(
    word_tokens
)

print(pair_counts)


def get_best_pair(
    pair_counts
):

    if not pair_counts:
        return None

    return pair_counts.most_common(1)[0][0]


best_pair = get_best_pair(
    pair_counts
)

print(best_pair)


def merge_pair(
    tokens,
    pair
):

    new_tokens = []

    i = 0

    while i < len(tokens):

        if (
            i < len(tokens) - 1
            and
            (tokens[i], tokens[i + 1])
            == pair
        ):

            merged = (
                tokens[i]
                +
                tokens[i + 1]
            )

            new_tokens.append(
                merged
            )

            i += 2

        else:

            new_tokens.append(
                tokens[i]
            )

            i += 1

    return new_tokens


tokens = [
    "l",
    "o",
    "w"
]

print(
    merge_pair(
        tokens,
        ("o", "w")
    )
)


def merge_all_words(
    word_tokens,
    pair
):

    for word in word_tokens:

        word_tokens[word] = merge_pair(
            word_tokens[word],
            pair
        )

merge_all_words(
    word_tokens,
    best_pair
)


def train_bpe(
    words,
    num_merges
):

    word_tokens = initialize_words(
        words
    )

    vocab = get_initial_vocab(
        words
    )

    merges = []

    for _ in range(num_merges):

        pair_counts = count_pairs(
            word_tokens
        )

        if not pair_counts:
            break

        best_pair = get_best_pair(
            pair_counts
        )

        merges.append(
            best_pair
        )

        new_token = (
            best_pair[0]
            +
            best_pair[1]
        )

        vocab.add(
            new_token
        )

        merge_all_words(
            word_tokens,
            best_pair
        )

    return vocab, merges


vocab, merges = train_bpe(
    words,
    num_merges=20
)

print("Vocabulary:")
print(vocab)

print()

print("Merges:")
for merge in merges:
    print(merge)



special_tokens = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3
}


token_to_id = {}

for token, idx in special_tokens.items():

    token_to_id[token] = idx


next_id = len(token_to_id)

for token in sorted(vocab):

    if token not in token_to_id:

        token_to_id[token] = next_id

        next_id += 1




def encode_word(
    word,
    merges
):

    tokens = list(word)

    for pair in merges:

        tokens = merge_pair(
            tokens,
            pair
        )

    return tokens


tokens = encode_word(
    "lower",
    merges
)

print(tokens)


def encode(
    text,
    merges,
    token_to_id
):

    words = text.split()

    ids = []

    for word in words:

        tokens = encode_word(
            word,
            merges
        )

        for token in tokens:

            token_id = token_to_id.get(
                token,
                token_to_id["<UNK>"]
            )

            ids.append(
                token_id
            )

    return ids


text = "low lower"

ids = encode(
    text,
    merges,
    token_to_id
)

print(ids)


id_to_token = {
    idx: token
    for token, idx
    in token_to_id.items()
}


def decode(
    ids,
    id_to_token
):

    tokens = []

    for idx in ids:

        token = id_to_token.get(
            idx,
            "<UNK>"
        )

        tokens.append(
            token
        )

    return "".join(tokens)


def preprocess(text):

    return text.replace(
        " ",
        "▁"
    )





class BPETokenizer:

    def __init__(
        self,
        vocab,
        merges
    ):

        self.merges = merges

        self.vocab = vocab

        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3
        }

        self.token_to_id = dict(
            self.special_tokens
        )

        next_id = len(
            self.token_to_id
        )

        for token in sorted(vocab):

            if token not in self.token_to_id:

                self.token_to_id[
                    token
                ] = next_id

                next_id += 1

        self.id_to_token = {
            idx: token
            for token, idx
            in self.token_to_id.items()
        }

    def encode(
        self,
        text,
        add_bos=False,
        add_eos=False
    ):

        text = text.replace(
            " ",
            "▁"
        )

        tokens = list(text)

        for pair in self.merges:

            tokens = merge_pair(
                tokens,
                pair
            )

        ids = []

        if add_bos:

            ids.append(
                self.token_to_id["<BOS>"]
            )

        for token in tokens:

            ids.append(
                self.token_to_id.get(
                    token,
                    self.token_to_id["<UNK>"]
                )
            )

        if add_eos:

            ids.append(
                self.token_to_id["<EOS>"]
            )

        return ids


    def decode(
        self,
        ids
    ):

        tokens = []

        for idx in ids:

            token = self.id_to_token.get(
                idx,
                "<UNK>"
            )

            if token in [
                "<PAD>",
                "<BOS>",
                "<EOS>"
            ]:

                continue

            tokens.append(token)

        text = "".join(tokens)

        text = text.replace(
            "▁",
            " "
        )

        return text


text = """
The transformer is a neural network architecture.
Transformers process sequences using attention.
Attention allows the model to understand relationships
between different tokens.
A language model predicts the next token.
GPT is a decoder-only transformer.
The model learns by predicting tokens from context.
Deep learning models require data and optimization.
The transformer architecture is powerful for language.
"""


words = text.lower().split()

vocab, merges = train_bpe(
    words,
    num_merges=50
)

tokenizer = BPETokenizer(
    vocab,
    merges
)

print("Vocabulary size:")
print(len(tokenizer.token_to_id))

print("\nMerges:")

for merge in tokenizer.merges:
    print(merge)


def test_tokenizer(text: str):
    ids = tokenizer.encode(
        text,
        add_bos=True,
        add_eos=True
    )

    print(ids)

    decoded = tokenizer.decode(
        ids
    )

    print(decoded)

# test_tokenizer(text = "low lower lowest")


tokens = tokenizer.encode(
    text.lower()
)

print(tokens)
print("Number of tokens:", len(tokens))


data = torch.tensor(
    tokens,
    dtype=torch.long
)

print(data.shape)

x = data[:-1]
y = data[1:]


def get_batch(
    data,
    batch_size,
    block_size,
    device
):

    starts = torch.randint(
        0,
        len(data) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        data[
            i:i + block_size
        ]
        for i in starts
    ])

    y = torch.stack([
        data[
            i + 1:i + block_size + 1
        ]
        for i in starts
    ])

    return (
        x.to(device),
        y.to(device)
    )

x, y = get_batch(
    data,
    batch_size=4,
    block_size=32,
    device=device
)