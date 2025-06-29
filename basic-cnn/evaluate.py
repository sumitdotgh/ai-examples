from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.utils import load_img, img_to_array # type: ignore
import numpy as np


if __name__ == "__main__":

    print("---Evaluating the model-----")

    # 1. Load the Model
    print("loading the model from file")
    model = load_model("letter_classification_model.keras")
    print(model)

    # 2. Preprocess the Input Image
    print("Perform pre-processing on the input image")
    img_path = "test_images/2.png"
    img = load_img(img_path, color_mode="grayscale", target_size=(28, 28))  # Assuming 28x28 input    
    img_array = img_to_array(img) / 255.0  # Normalize pixel values (if needed)
    img_array = img_array.reshape(1, 28, 28, 1)

    # 3. Make the Prediction
    print("Making prediction")
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    print(predicted_class)

    # Assuming you have a dictionary mapping class indices to letters    
    class_labels = {0: 'Zero', 1: 'One', 2: 'Two'}
    predicted_letter = class_labels[predicted_class] # type: ignore    
    print("*********** OUTPUT ****************")
    print(f"Predicted letter: {predicted_letter}")
    print("**********************************")