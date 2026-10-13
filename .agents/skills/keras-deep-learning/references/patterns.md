# Keras 3 & Deep Learning Reference

This reference documents neural network design and serialization standards used across `basic-cnn`, `tiny-moe-based-model`, and `basic-peft`.

## Custom Layer Serialization
Every custom layer MUST be serializable for saving to `.keras`:

```python
import tensorflow as tf
from keras import layers, saving

@saving.register_keras_serializable()
class CustomLayer(layers.Layer):
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
            name="kernel"
        )
        super().build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
```

## Training Loop & Callbacks
Always use early stopping and dynamic learning rate reduction:

```python
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
]

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20,
    batch_size=64,
    callbacks=callbacks
)

# Save in modern .keras format
model.save("model.keras")
```
