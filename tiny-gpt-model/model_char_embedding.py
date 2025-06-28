import numpy as np
import tensorflow as tf
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore


def encode(text):
    """Method to encode to text"""
    return [stoi[c] for c in text]

def decode(indices):
    """Method to do the decoding of the text"""
    return ''.join([itos[i] for i in indices])

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
    """Method to define model architecture"""

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

def prepare_input(text, seq_length=8):
    """Method to prepare input text"""
    token_ids = encode(text)
    if len(token_ids) < seq_length:
        token_ids = [0] * (seq_length - len(token_ids)) + token_ids
    else:
        token_ids = token_ids[-seq_length:]
    return np.array([token_ids])

def generate_text(model, input_text,num_generate=20):
    """Method to generate the text based on model prediction"""
    model_input = encode(input_text)
    for _ in range(num_generate):
        x = prepare_input(input_text, seq_length)  # pad to seq length
        preds = model.predict(x, verbose=0)[0, -1]
        next_id = np.argmax(preds)
        model_input.append(next_id) # type: ignore        
    return decode(model_input)

if __name__ == "__main__":
    
    print("************************************************")    
    print("************** Tiny GPT Model ******************")

    # Tiny text corpus    
    corpus = "hello world how are you today what is up hello world again fine thank you"

    # Character-level tokenizer
    chars = sorted(list(set(corpus)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    ## Fetch data corpus
    data = encode(corpus)    

    ## Create input-output sequences
    seq_length = 8
    inputs = []
    targets = []

    for i in range(len(data) - seq_length):
        inputs.append(data[i:i+seq_length])
        targets.append(data[i+1:i+seq_length+1])

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
    print(generate_text(model, "hello "))
    print("************************************************")