import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import numpy as np
import json
import os

# Path to your saved model folder
MODEL_DIR = "./distilbert-lora-peft-tf-cls"

# --------------------------
# Prediction function
# --------------------------
def predict_single(text):
    enc = tokenizer(
        text,
        return_tensors="tf",
        truncation=True,
        padding="max_length",
        max_length=64
    )
    enc = {k: tf.cast(v, tf.int32) for k, v in enc.items()}
    out = model(**enc, training=False)
    logits = out.logits
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    pred_idx = int(np.argmax(probs))
    return inv_label_map[pred_idx], float(probs[pred_idx])

if __name__ == "__main__":

    # --------------------------
    # Load tokenizer & label map
    # --------------------------
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)

    label_map_path = os.path.join(MODEL_DIR, "label_map.json")
    with open(label_map_path, "r") as f:
        label_map = json.load(f)

    inv_label_map = {v: k for k, v in label_map.items()}

    # --------------------------
    # Load trained model
    # --------------------------
    model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

    # --------------------------
    # Example usage
    # --------------------------    
    samples = [
    # News style
    "Jon: Have you seen the new OpenAI model release?\nSumit: Yes, it just came out yesterday.",
    "Jon: Did you hear about Microsoft’s new AI-powered search?\nSumit: Yes, it’s trending right now.",

    # Opinion style
    "Jon: I think LLaMA 3 is better than GPT-4.\nSumit: Not sure, GPT-4 seems more consistent.",
    "Jon: Fine-tuning is a waste of time compared to prompt engineering.\nSumit: I disagree, fine-tuning has its place.",

    # Comparison style
    "Jon: Which is faster, Gemini or GPT?\nSumit: Gemini feels quicker, but GPT is more reliable.",
    "Jon: Should we use LangChain or LlamaIndex for RAG?\nSumit: Both work, but LangChain has better integration.",

    # Mixed / edge cases
    "Jon: Did you test the new Claude API?\nSumit: Yes, it’s impressive.",
    "Jon: I think smaller models are more efficient for edge devices.\nSumit: True, they require fewer resources."
    ]

    for sample in samples:
        print("------------------")
        lbl, conf = predict_single(sample)
        print(f"Sample: {sample}")
        print(f"Prediction: {lbl} (confidence={conf:.4f})")
        print("------------------")
