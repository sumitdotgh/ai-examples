"""Tiny Transformer baseline used to contrast HOPE-style models.

Architecture (high level):
  - Input  : sequence of token ids (one id per word position).
  - Embed  : look up a dense vector for each token id (shared table).
  - Blocks : repeat N times:
      * Multi-head self-attention (each position looks at every other).
      * Add & LayerNorm (residual connection).
      * Position-wise feed-forward MLP.
      * Add & LayerNorm again.
  - Output : a Dense layer that predicts a distribution over the vocabulary
             for every position (next-token prediction).
"""

from typing import Optional

import tensorflow as tf


def build_tiny_transformer(
    vocab_size: int,
    seq_len: int,
    d_model: int = 64,
    num_heads: int = 2,
    ff_dim: int = 128,
    num_layers: int = 2,
    dropout_rate: float = 0.1,
    learning_rate: float = 2e-3,
) -> tf.keras.Model:
    """Minimal encoder-only Transformer that predicts the next token."""

    # 1. Input layer: a batch of integer token ids with fixed length `seq_len`.
    inputs = tf.keras.Input(shape=(seq_len,), dtype="int32")

    # 2. Embedding: map each token id to a dense `d_model`-dimensional vector.
    #    This is like learning a lookup table of word meanings.
    x = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)

    # 3. Stack several identical Transformer encoder blocks.
    for _ in range(num_layers):
        # 3a. Multi-head self-attention:
        #     - Queries, keys, and values all come from `x`.
        #     - Each position can attend to every other position in the sequence.
        attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(
            x, x
        )
        # 3b. Dropout for regularization, then Add & LayerNorm (residual connection).
        attn = tf.keras.layers.Dropout(dropout_rate)(attn)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)

        # 3c. Position-wise feed-forward network:
        #     - Two Dense layers applied independently at each position.
        #     - Expands to `ff_dim`, then compresses back to `d_model`.
        ffn = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
        ffn = tf.keras.layers.Dense(d_model)(ffn)
        ffn = tf.keras.layers.Dropout(dropout_rate)(ffn)
        # 3d. Second residual + LayerNorm around the feed-forward part.
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn)

    # 4. Final Dense layer: for every position, predict scores over the vocab.
    logits = tf.keras.layers.Dense(vocab_size)(x)
    model = tf.keras.Model(inputs=inputs, outputs=logits, name="tiny_transformer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # fast learner
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model

