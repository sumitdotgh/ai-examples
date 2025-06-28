import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore

class SimpleSelfAttention(layers.Layer):
    """Class to perform self attention"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ln = layers.LayerNormalization()
        self.add = layers.Add()

    def call(self, x):
        """Method to call self attention"""
        attn_mask = tf.linalg.band_part(tf.ones((x.shape[1], x.shape[1])), -1, 0)
        attn_output = self.attn(x, x, attention_mask=attn_mask)
        return self.ln(self.add([x, attn_output]))

def get_gpt_model(vocab_size, seq_length, embed_dim=32, num_heads=2, ff_dim=64):

    # Keras input with sequence length 8
    inputs = keras.Input(shape=(seq_length,))
    x = layers.Embedding(vocab_size, embed_dim)(inputs)

    # Add positional embeddings
    pos_embed = layers.Embedding(input_dim=seq_length, output_dim=embed_dim)(tf.range(start=0, limit=seq_length))
    x = x + pos_embed

    # Transformer block
    x = SimpleSelfAttention(embed_dim, num_heads)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dense(embed_dim)(x)

    # Output logits
    logits = layers.Dense(vocab_size)(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=logits)

    return model

def generate_text(model, input_text, tokenizer, seq_length, num_words=10):
    """Method to generate next text based on input text"""
    result = input_text.split()
    for _ in range(num_words):
        encoded = tokenizer.texts_to_sequences([" ".join(result)])[0]
        padded = pad_sequences([encoded], maxlen=seq_length, padding='pre')
        preds = model.predict(padded, verbose=0)[0, -1]
        predicted_id = np.argmax(preds)
        for word, index in tokenizer.word_index.items():
            if index == predicted_id:
                result.append(word)
                break
    return " ".join(result)

if __name__ == "__main__":
    
    print("************************************************")    
    print("************** Tiny GPT Model ******************")

    # Tiny text corpus    
    corpus = [
    "hello world how are you",
    "how are you doing today",
    "hello world again",
    "fine thank you"
    ]    

    # Use Keras Tokenizer (word level)
    tokenizer = Tokenizer(oov_token="[OOV]", lower=True)
    tokenizer.fit_on_texts(corpus)

    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1  # +1 for padding
    print("Vocab:", word_index)    

    ## Create input-output sequences
    # Create input and target sequences
    seqs = []
    for line in corpus:
        tokens = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(tokens)):
            input_seq = tokens[:i]
            target_seq = tokens[1:i+1]
            seqs.append((input_seq, target_seq))

    # Pad sequences to fixed length
    seq_length = 6  # you can change this
    inputs = []
    targets = []

    for input_seq, target_seq in seqs:
        inputs.append(pad_sequences([input_seq], maxlen=seq_length, padding='pre')[0])
        targets.append(pad_sequences([target_seq], maxlen=seq_length, padding='pre')[0])

    inputs = np.array(inputs)
    targets = np.array(targets)

    print("----inputs-----")
    print(inputs)
    print("----targets----")
    print(targets)

    ## Fetch GPT model    
    print("----model----")
    model = get_gpt_model(vocab_size, seq_length)
    print(model)

    ## Define the loss function
    print("----loss----")
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    print(loss_fn)

    ## Compile the model with optimizer and loss function
    print("-----compile model----")
    model.compile(optimizer="adam", loss=loss_fn)    

    print("----model summary----")
    model.summary()    

    ## Train the model 
    print("----train model-------")
    model.fit(inputs, targets, epochs=100, batch_size=2)

    ## Inference 
    print("-----generated text-----")
    print(generate_text(model, "hello world ", tokenizer, seq_length))
    print("************************************************")

