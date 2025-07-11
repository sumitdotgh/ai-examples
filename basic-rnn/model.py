from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import numpy as np


def get_rnn_model(vocab_size, seq_length):

    model = models.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=seq_length),
        layers.SimpleRNN(64, return_sequences=False),
        layers.Dense(64, activation='relu'),
        layers.Dense(vocab_size, activation='softmax')
    ])

    return model

def predict_next(model, tokenizer, text, seq_length=4):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=seq_length, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_id = np.argmax(predicted_probs)
    for word, index in tokenizer.word_index.items():
        if index == predicted_id:
            return word
    return ""

if __name__ == "__main__":
    
    print("************************************************")    
    print("************** Basic RNN ***********************")

    corpus = [
        "hello world how are you",
        "how are you doing today",
        "hello world again",
        "fine thank you"
    ]

    tokenizer = Tokenizer(oov_token="[OOV]", lower=True)
    tokenizer.fit_on_texts(corpus)

    sequences = []
    seq_length = 4

    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            input_seq = token_list[:i]
            input_seq = pad_sequences([input_seq], maxlen=seq_length, padding='pre')[0]
            target = token_list[i]
            sequences.append((input_seq, target))

    X = np.array([x for x, y in sequences])
    y = np.array([y for x, y in sequences])

    vocab_size = len(tokenizer.word_index) + 1

    model = get_rnn_model(vocab_size,seq_length)
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    model.fit(X, y, epochs=200, verbose=1)