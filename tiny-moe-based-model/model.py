import numpy as np
import pickle
import tensorflow as tf
from keras import layers, Model, Input, Sequential, saving
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# ==========================
# Configs
# ==========================
vocab_size = 1000      # Example vocab
embed_dim = 32
num_experts = 4
hidden_dim = 64

# ==========================
# Gating Network
# ==========================
@saving.register_keras_serializable()
class GatingNetwork(layers.Layer):
    def __init__(self, num_experts, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.dense = layers.Dense(num_experts, activation="softmax")

    def call(self, x):
        return self.dense(x)

    def get_config(self):
        config = super().get_config()
        config.update({"num_experts": self.num_experts})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ==========================
# Mixture of Experts Layer
# ==========================
@saving.register_keras_serializable()
class MoELayer(layers.Layer):
    def __init__(self, num_experts, hidden_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gating = GatingNetwork(num_experts)
        self.experts = []  # Will be created in build()

    def build(self, input_shape):
        self.experts = []
        for _ in range(self.num_experts):
            expert = Sequential([
                layers.Dense(self.hidden_dim, activation="relu"),
                layers.Dense(self.output_dim)
            ])
            expert.build(input_shape)
            self.experts.append(expert)
        super().build(input_shape)

    def _compute_expert_outputs(self, x):
        """
        Passes the input through all experts and stacks their outputs.
        Output shape: (batch, num_experts, output_dim)
        """
        expert_outputs = [expert(x) for expert in self.experts]
        return tf.stack(expert_outputs, axis=1)
    
    def call(self, x):
        gate_values = self.gating(x)  # (batch, num_experts)
        expert_outputs = self._compute_expert_outputs(x)  # (batch, num_experts, output_dim)
        gate_values = tf.expand_dims(gate_values, axis=-1)  # (batch, num_experts, 1)
        return tf.reduce_sum(gate_values * expert_outputs, axis=1)  # (batch, output_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_experts": self.num_experts,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":

    # ==========================
    # Step 1: Tokenize the corpus
    # ==========================
    corpus = [
        "hello world how are you",
        "how are you doing today",
        "hello world again",
        "fine thank you"
    ]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    word_index = tokenizer.word_index
    index_word = {v: k for k, v in word_index.items()}
    vocab_size = len(word_index) + 1  # +1 for padding

    print("Vocabulary:", word_index)

    # ==========================
    # Step 2: Generate input-output pairs
    # ==========================
    input_sequences = []
    target_tokens = []

    for line in corpus:
        tokens = tokenizer.texts_to_sequences([line])[0]  # e.g., [1, 2, 3, 4]
        for i in range(1, len(tokens)):
            input_seq = tokens[:i]
            target = tokens[i]
            input_sequences.append(input_seq)
            target_tokens.append(target)

    # Pad sequences to the same length
    max_seq_len = max(len(seq) for seq in input_sequences)
    X = pad_sequences(input_sequences, maxlen=max_seq_len)
    y = np.array(target_tokens)

    print("\nSample input-output:")
    for i in range(len(X)):
        input_words = [index_word.get(id, '') for id in X[i]]
        print(f"Input: {input_words} → Target: {index_word[y[i]]}")
        

    # ==========================
    # Step 3: Build MoE Language Model
    # ==========================
    inputs = Input(shape=(max_seq_len,))
    x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)
    x = layers.GlobalAveragePooling1D()(x)  # Reduce sequence dimension
    x = MoELayer(num_experts=num_experts, hidden_dim=hidden_dim, output_dim=embed_dim)(x)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # ==========================
    # Step 3: Training
    # ==========================    
    model.fit(X, y, epochs=100, batch_size=8) 

    # ==========================
    # Step 4: Save to .keras format
    # ==========================
    
    # Save the Max Seq length
    with open("saved_model/max_seq_len.txt", "w") as f:
        f.write(str(max_seq_len))

    # Save model as keras format
    model.save("saved_model/tiny_moe_model.keras")

    # Save the tokenizer
    with open("saved_model/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)