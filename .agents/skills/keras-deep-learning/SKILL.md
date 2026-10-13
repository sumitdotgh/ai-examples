---
name: keras-deep-learning
description: >-
  Guides designing, training, evaluating, serializing, and serving neural networks using Keras 3 and TensorFlow. Use this skill whenever building or updating CNN, RNN, MoE, Transformer, PEFT/LoRA, or custom deep learning architectures.
---

# Keras Deep Learning Skill

Use this skill to implement clean, reproducible deep learning models conforming to the repository standards.

## Workflow

1. **Python & Environment**:
   - Verify Python environment is Python 3.10-3.11 compatible with TensorFlow wheels.
   - Configure Poetry dependencies with `keras>=3.0.0` and `tensorflow>=2.16.0`.

2. **Custom Layers**:
   - Annotate all custom layers with `@saving.register_keras_serializable()`.
   - Implement `build()`, `call()`, `get_config()`, and `from_config()`.

3. **Training & Callbacks**:
   - Use `EarlyStopping(monitor="val_loss", restore_best_weights=True)`.
   - Use `ReduceLROnPlateau(monitor="val_loss", factor=0.5)`.

4. **Model Saving**:
   - Save models to modern `.keras` zip format (e.g. `letter_classification_model.keras`).

5. **Separation**:
   - Separate training in `model.py` from inference evaluation in `evaluate.py` or `serve.py`.

Review [references/patterns.md](./references/patterns.md) for custom layer and callback implementations.
