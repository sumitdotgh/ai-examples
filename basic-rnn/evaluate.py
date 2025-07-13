from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import numpy as np


def predict_next_word(model,input_text):
    """Predicts the next word given a partial sequence"""
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    token_list = pad_sequences([token_list], maxlen=seq_length, padding='pre')
    
    prediction = model.predict(token_list, verbose=0)
    predicted_id = np.argmax(prediction)
    
    predicted_word = idx_to_word.get(predicted_id, "[UNK]")
    return predicted_word


if __name__ == "__main__":

    print("*********** Evaluating the model *****************") 

    # 1. Load the model
    loaded_model = load_model("next_word_prediction_model.keras")
    print(loaded_model)
    
    # Recreate the same tokenizer and corpus used for training
    corpus = [
        "hello world how are you",
        "how are you doing today",
        "hello world again",
        "fine thank you"
    ]

    tokenizer = Tokenizer(oov_token="[OOV]", lower=True)
    tokenizer.fit_on_texts(corpus)

    # Parameters
    seq_length = 4
    vocab_size = len(tokenizer.word_index) + 1
    idx_to_word = {v: k for k, v in tokenizer.word_index.items()}

    print("*********** OUTPUT ****************")    
    while True:
        seed = input("\nEnter a 1-4 word phrase (or 'exit' to quit): ").strip().lower()
        if seed == "exit":
            break
        next_word = predict_next_word(loaded_model,seed)
        print(f"🧠 Predicted next word: {next_word}")
    print("**********************************")
    
