import numpy as np


sentences = [
    "i love machine learning",
    "i love deep learning",
    "machine learning is powerful",
    "deep learning is interesting",
    "i study machine learning"
]


tokenized_sentences = [sentence.lower().split() for sentence in sentences]
vocab = sorted(set(word for sentence in tokenized_sentences for word in sentence))
word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for word, i in word_to_id.items()}
encoded_sentences = [[word_to_id[word] for word in sentence] for sentence in tokenized_sentences]


X = [sentence[:-1] for sentence in encoded_sentences]
Y = [sentence[1:] for sentence in encoded_sentences]


vocab_size = len(vocab)
print(vocab_size)
hidden_dim = 16             # size of hidden state
learning_rate = 0.1


# initialize model parameter (weights and biases)
np.random.seed(42)

W_xh = np.random.randn(hidden_dim, vocab_size) * 0.01   # input to hidden
W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01   # hidden to hidden
W_hy = np.random.randn(vocab_size, hidden_dim) * 0.01   # hidden to output
b_h = np.zeros((hidden_dim, 1))     # hidden bias
b_y = np.zeros((vocab_size, 1))     # output bias


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


epochs = 200

for epoch in range(epochs):
    loss = 0

    # train using one sentence at a time
    for seq_x, seq_y in zip(X, Y):
        T = len(seq_x)

        # dictionary to store activation at each time step for the backward pass
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.zeros((hidden_dim, 1))      # initial hidden state (h_minus_1)

        #forward pass
        for t in range(T):
            # one hot encode the input token
            xs[t] = np.zeros((vocab_size, 1))
            xs[t][seq_x[t]] = 1

            # compute hidden state: tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
            hs[t] = np.tanh(np.dot(W_hh, hs[t-1]) + np.dot(W_xh, xs[t]) + b_h)

            # compute unnormalized log probabilities (logits) and apply Softmax
            ys[t] = np.dot(W_hy, hs[t]) + b_y
            ps[t] = softmax(ys[t])

            # cross-entropy loss
            loss -= np.log(ps[t][seq_y[t], 0] + 1e-15)

        # Back-propagation through time (BTT)
        # initialize gradients to 0
        dW_xh, dW_hh, dW_hy = np.zeros_like(W_xh), np.zeros_like(W_hh), np.zeros_like(W_hy)
        db_h, db_y = np.zeros_like(b_h), np.zeros_like(b_y)
        dh_next = np.zeros_like(hs[0])

        # look backwards through time steps
        for t in reversed(range(T)):

            # 1. gradient of loss with respect to output logits
            dy = np.copy(ps[t])
            dy[seq_y[t]] -= 1       # derivative of cross-entropy + softmax

            # 2. gradients for output parameter
            dW_hy += np.dot(dy, hs[t].T)
            db_y += dy

            # 3. gradient flowing back into the hidden state
            dh = np.dot(W_hy.T, dy) + dh_next

            # 4. Backprop through the tanh non-linearity
            dh_raw = (1 - hs[t] ** 2) * dh

            # 5. gradients for RNN cell parameters
            dW_xh += np.dot(dh_raw, xs[t].T)
            dW_hh += np.dot(dh_raw, hs[t-1].T)
            db_h += dh_raw

            # pass gradient back to the previous time step
            dh_next = np.dot(W_hh.T, dh_raw)


        # parameter updates (gradient descent)
        W_xh -= learning_rate * dW_xh
        W_hh -= learning_rate * dW_hh
        W_hy -= learning_rate * dW_hy
        b_h -= learning_rate * db_h
        b_y -= learning_rate * db_y

    if (epoch + 1) % 40 == 0:
        print(f"Epoch: {(epoch + 1)/epochs} Loss: {loss:.4f}")


print("Training complete")


# Inference / Text prediction
def predict_next_word(input_sentence: str):

    words = input_sentence.lower().split()
    h = np.zeros((hidden_dim, 1))

    # process the sequence upto the last word
    for word in words:
        if word not in word_to_id:
            return f"{word} is not in vocabulary"

        x = np.zeros((vocab_size, 1))
        x[word_to_id[word]] = 1
        h = np.tanh(np.dot(W_hh, h) + np.dot(W_xh, x) + b_h)

    # calculate output for the last hidden state
    y = np.dot(W_hy, h) + b_y
    p = softmax(y)

    predicted_id = np.argmax(p)
    return id_to_word[predicted_id]

test_phrases = ["i love", "machine learning", "i study"]
for phrase in test_phrases:
    pred = predict_next_word(phrase)
    print(f"Input: {phrase} -> Predicted next word: {pred}")









