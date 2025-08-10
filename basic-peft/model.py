import os
import json
import numpy as np
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
from data import CORPUS  # Corpus with string labels

SAVE_DIR = "distilbert-lora-peft-tf-cls"

# --------------------------
# LoRA adapter layer
# --------------------------
class LoRALayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, rank=4, alpha=32, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.lora_A = self.add_weight(
            shape=(input_dim, rank),
            initializer="random_normal",
            trainable=True,
            name="lora_A"
        )
        self.lora_B = self.add_weight(
            shape=(rank, input_dim),
            initializer="zeros",
            trainable=True,
            name="lora_B"
        )

    def call(self, inputs):
        # inputs: [batch, seq_len, hidden_dim]
        # apply LoRA: (x @ A @ B) * scale
        return tf.einsum('bij,jk->bik', inputs, tf.matmul(self.lora_A, self.lora_B)) * self.scale

# --------------------------
# Prepare data: labels -> ints, tokenize, create tf.data.Dataset
# --------------------------
# 1) Extract texts and string labels
texts = [row["text"] for row in CORPUS]
labels_str = [row["label"] for row in CORPUS]

# 2) Create label map (string -> int) and inverse map
unique_labels = sorted(list(set(labels_str)))
label_map = {lab: idx for idx, lab in enumerate(unique_labels)}
inv_label_map = {v: k for k, v in label_map.items()}
num_labels = len(unique_labels)
print("===== Label map =====")
print("Label map:", label_map)

# ----------------------------
# 3) Convert labels to integers
# -----------------------------
labels = np.array([label_map[s] for s in labels_str], dtype=np.int32)

# --------------------------
# 4) Tokenizer + encodings
# --------------------------
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
encodings = tokenizer(
    texts,
    truncation=True,
    padding="max_length",  # fixed-length for batching; change to True variable-length if preferred
    max_length=64,
    return_tensors="np"
)

# ------------------------------------
# Convert numpy arrays to int32 for TF
# ------------------------------------
input_ids = encodings["input_ids"].astype(np.int32)
attention_mask = encodings["attention_mask"].astype(np.int32)

# -------------------------
# 5) Build tf.data.Dataset
# --------------------------
dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": input_ids, "attention_mask": attention_mask},
    labels
))

# --------------------------
# Shuffle + split
# --------------------------
dataset = dataset.shuffle(buffer_size=len(CORPUS), seed=42)
train_size = int(0.8 * len(CORPUS))
train_ds = dataset.take(train_size).batch(8)
val_ds = dataset.skip(train_size).batch(8)

# --------------------------------------
# Load model and inject LoRA adapters into last 2 layers
# ---------------------------------------
model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Freeze base transformer layers (optional)
for layer in model.distilbert.transformer.layer:
    layer.trainable = False

# Inject LoRA into the last 2 transformer layers (FFN lin2)
for i in [-2, -1]:
    transformer_layer = model.distilbert.transformer.layer[i]
    ffn = transformer_layer.ffn

    # get hidden size from lin2 kernel shape (lin2.kernel shape: [hidden_dim_ffn, hidden_dim])
    input_dim = int(ffn.lin2.kernel.shape[0])  # cast to int to be safe
    lora_adapter = LoRALayer(input_dim)

    original_lin2_call = ffn.lin2.call

    # define wrapper that adds LoRA output to input before original lin2 call
    def new_call(x, *args, original_lin2_call=original_lin2_call, lora_adapter=lora_adapter, **kwargs):
        # x shape: (batch, seq_len, hidden_dim)
        x = x + lora_adapter(x)
        return original_lin2_call(x, *args, **kwargs)

    # patch the call method and make trainable
    ffn.lin2.call = new_call
    ffn.lin2.trainable = True

# Make classification head trainable
model.classifier.trainable = True

# --------------------------
# Compile and train
# --------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

# Train
model.fit(train_ds, validation_data=val_ds, epochs=3)

# --------------------------
# Save model & tokenizer
# --------------------------
if os.path.exists(SAVE_DIR):
    # optional: remove prior directory to avoid overwrite issues
    import shutil
    shutil.rmtree(SAVE_DIR)

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

label_map_path = os.path.join(SAVE_DIR, "label_map.json")
with open(label_map_path, "w") as f:
    json.dump(label_map, f, indent=2)

print(f"Saved model, tokenizer, and label map to: {SAVE_DIR}")