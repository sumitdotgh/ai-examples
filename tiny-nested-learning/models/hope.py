"""Tiny HOPE-style continuum memory model.

High-level idea:
  - Classic RNNs keep a single hidden state that is updated every step.
  - HOPE-style cells instead keep several hidden states that update at
    different speeds (fast / medium / slow).
  - Fast states react quickly but forget quickly, slow states move slowly
    but remember information for a long time.
  - A small "controller" network decides *how much* to update each state
    at every time step, based on the current input and the past states.
"""

from typing import List

import tensorflow as tf


class HopeCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN cell that keeps multiple memories (fast/medium/slow) and blends them
    using learnable update rates, echoing the Continuum Memory System idea.
    """

    def __init__(self, units: int, update_rates: List[float], **kwargs):
        super().__init__(**kwargs)
        # Size of each memory vector (we will have one vector per timescale).
        self.units = units
        # Base update rates, e.g. [0.8, 0.2, 0.05] for fast/medium/slow.
        self.update_rates = update_rates
        # Controller proposes a new "candidate" memory based on inputs + old states.
        self.controller = tf.keras.layers.Dense(units, activation="tanh")
        # Rate adapter slightly adjusts the base rate for each timescale using context.
        self.rate_adapter = tf.keras.layers.Dense(
            len(update_rates), activation="sigmoid"
        )

    @property
    def state_size(self):
        # We keep one hidden state vector for each timescale (fast/medium/slow).
        return [self.units for _ in self.update_rates]

    @property
    def output_size(self):
        return self.units

    def call(self, inputs, states):
        """
        One RNN step:
          - `inputs` is the embedding at the current time step.
          - `states` is a list of hidden states [fast_state, medium_state, slow_state].
        We:
          1. Concatenate inputs + all states.
          2. Use the controller to propose a new memory vector.
          3. Use rate_adapter to pick an update weight for each timescale.
          4. Exponentially move each state toward the proposed memory.
        """

        # 1. Combine current input with all memory tracks into a single feature vector.
        concat = tf.concat([inputs] + list(states), axis=-1)

        # 2. Controller proposes a shared "candidate" content that all memories can move toward.
        proposal = self.controller(concat)

        # 3. Rate adapter produces a small vector (0..1 per timescale)
        #    telling us how much to update the fast/medium/slow memories right now.
        rate_adjust = self.rate_adapter(concat)

        # 4. Update each memory using a dynamic exponential moving average (EMA).
        new_states = []
        for idx, (state, base_rate) in enumerate(zip(states, self.update_rates)):
            # Dynamic rate is close to base_rate, but can be increased/decreased slightly.
            dynamic_rate = base_rate + (1.0 - base_rate) * rate_adjust[:, idx : idx + 1]
            # EMA-style update:
            #   new_state = (1 - rate) * old_state + rate * proposal
            # Fast memories (high base_rate) change quickly, slow memories change slowly.
            updated = (1.0 - dynamic_rate) * state + dynamic_rate * proposal
            new_states.append(updated)

        # 5. Output is the mean of fast/medium/slow memories.
        #    This gives a single feature vector for the RNN output at this time step.
        output = tf.reduce_mean(tf.stack(new_states, axis=0), axis=0)
        return output, new_states


def build_tiny_hope(
    vocab_size: int,
    seq_len: int,
    units: int = 96,
    update_rates: List[float] | None = None,
    dropout_rate: float = 0.0,
    learning_rate: float = 1.5e-3,
) -> tf.keras.Model:
    """Single-layer HOPE-style network with configurable memory speeds."""

    if update_rates is None:
        update_rates = [0.8, 0.2, 0.05]  # fast, medium, and slow memories

    inputs = tf.keras.Input(shape=(seq_len,), dtype="int32")
    x = tf.keras.layers.Embedding(vocab_size, units)(inputs)  # shared embedding table
    hope_block = tf.keras.layers.RNN(
        HopeCell(units, update_rates), return_sequences=True, name="hope_cms"
    )
    x = hope_block(x)  # run through CMS cell
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)  # stabilize activations
    logits = tf.keras.layers.Dense(vocab_size)(x)  # predict next token

    model = tf.keras.Model(inputs=inputs, outputs=logits, name="tiny_hope")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # slightly smaller lr
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model

