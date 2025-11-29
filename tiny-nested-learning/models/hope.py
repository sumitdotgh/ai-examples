"""
Tiny HOPE-style continuum memory model.

High-level idea:
  - Transformers update one shared memory (weights) → prone to catastrophic forgetting.
  - HOPE-style RNNs maintain *multiple memories* that update at different speeds:
        Fast   → reacts quickly, forgets quickly
        Medium → balanced
        Slow   → very stable, preserves long-term knowledge
  - A controller proposes a new "candidate" memory.
  - A rate adapter adjusts how strongly each memory moves toward it.
  - Slow memory must remain slow → core idea of Nested Learning.

This implementation is tuned to demonstrate *better retention* than a Transformer.
"""

from typing import List
import tensorflow as tf


# =======================================================================
# HOPE CELL (Continuum Memory System)
# =======================================================================

class HopeCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN cell that maintains multiple memories (fast, medium, slow)
    and blends them using a controlled exponential moving average.

    This version includes:
      - Base update rates (fast/medium/slow)
      - Rate adapter network with CLAMPING to avoid overwriting slow memory
      - Strong retention behavior for continual learning demos
    """

    def __init__(self, units: int, update_rates: List[float], **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.update_rates = update_rates  # e.g., [0.6, 0.3, 0.02]

        # Controller creates a proposal vector for all memories
        self.controller = tf.keras.layers.Dense(units, activation="tanh")

        # Rate adapter slightly adjusts update strength for fast/medium memories
        self.rate_adapter = tf.keras.layers.Dense(len(update_rates), activation="sigmoid")

    @property
    def state_size(self):
        # One memory vector per timescale
        return [self.units for _ in self.update_rates]

    @property
    def output_size(self):
        return self.units

    def call(self, inputs, states):
        """
        One timestep of HOPE:
          1. Combine input & memories
          2. Controller proposes new content
          3. Rate adapter yields context-sensitive rates
          4. Slow memory is protected (dynamic_rate = base_rate)
          5. Fast + medium memories update with clamped dynamic_rate
        """

        # Combine input & memory tracks
        concat = tf.concat([inputs] + list(states), axis=-1)

        # Step 1: Controller proposal
        proposal = self.controller(concat)

        # Step 2: Contextual adjustment of update strength
        rate_adjust = self.rate_adapter(concat)  # values in (0,1)

        new_states = []

        for idx, (state, base_rate) in enumerate(zip(states, self.update_rates)):

            # -------------------------------------------
            # SLOW MEMORY PROTECTED: NO ACCIDENTAL SPEED-UP
            # -------------------------------------------
            if idx == 2:  # slow memory index
                dynamic_rate = base_rate  # fixed slow update rate (e.g., 0.02)

            else:
                # Allow dynamic adjustment but CLAMP so rates never explode
                dynamic_rate = base_rate + (1.0 - base_rate) * rate_adjust[:, idx:idx+1]

                # Prevent dynamic rates from overpowering base_rate
                dynamic_rate = tf.clip_by_value(
                    dynamic_rate,
                    clip_value_min=0.0,
                    clip_value_max=base_rate + 0.05  # e.g., fast: 0.65 max, med: 0.35 max
                )

            # Exponential Moving Average update (Nested Learning core idea)
            updated = (1.0 - dynamic_rate) * state + dynamic_rate * proposal
            new_states.append(updated)

        # Output = average of all memories (fast, medium, slow)
        output = tf.reduce_mean(tf.stack(new_states, axis=0), axis=0)
        return output, new_states


# =======================================================================
# HOPE MODEL WRAPPER
# =======================================================================

def build_tiny_hope(
    vocab_size: int,
    seq_len: int,
    units: int = 96,
    update_rates: List[float] | None = None,
    dropout_rate: float = 0.0,
    learning_rate: float = 7e-4,
) -> tf.keras.Model:
    """
    Builds a 1-layer HOPE model with controlled multi-timescale memory.

    Defaults:
      update_rates = [0.6, 0.3, 0.02] → strong continual learning performance
    """

    if update_rates is None:
        update_rates = [0.6, 0.3, 0.02]  # tuned for retention demo

    inputs = tf.keras.Input(shape=(seq_len,), dtype="int32")

    # Shared embedding
    x = tf.keras.layers.Embedding(vocab_size, units)(inputs)

    # Continuum Memory System block
    hope_block = tf.keras.layers.RNN(
        HopeCell(units, update_rates),
        return_sequences=True,
        name="hope_cms"
    )
    x = hope_block(x)

    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Stabilize output
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    # Predict next token
    logits = tf.keras.layers.Dense(vocab_size)(x)

    # Compile
    model = tf.keras.Model(inputs, logits, name="tiny_hope")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model
