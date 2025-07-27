import pickle
import numpy as np
from tensorflow.keras.utils import pad_sequences
from keras.models import load_model
from model import MoELayer, GatingNetwork

def predict_next(model,text):
    seq = tokenizer.texts_to_sequences([text])[0]
    padded = pad_sequences([seq], maxlen=max_seq_len)
    preds = model.predict(padded, verbose=0)[0]
    top_idx = np.argmax(preds)
    return index_word.get(top_idx, "<UNK>")


if __name__ == "__main__":

    print("============ Evaluating the model ===========")

    # =============================
    # Step 1: Load Tokenizer
    # =============================
    with open("saved_model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    word_index = tokenizer.word_index
    index_word = {v: k for k, v in word_index.items()}
    vocab_size = len(word_index) + 1

    # =============================
    # Step 2: Load Max Seq length
    # =============================
    max_seq_len = 0
    with open("saved_model/max_seq_len.txt", "r") as f:
        max_seq_len = int(f.read()) # Must match training value!    

    # ==========================
    # Step 3: Load model from file
    # ==========================
    loaded_model = load_model("saved_model/tiny_moe_model.keras", custom_objects={
    "MoELayer": MoELayer,
    "GatingNetwork": GatingNetwork
    })
    
    print("Model loaded successfully")

    # ==========================
    # Step 4: Verify prediction
    # ==========================
    print("\n Predictions:")
    test_phrases = [
        "hello world",
        "how are",
        "thank"
    ]

    for phrase in test_phrases:
        next_word = predict_next(loaded_model,phrase)
        print(f"Input: '{phrase}' → Next word: '{next_word}'")