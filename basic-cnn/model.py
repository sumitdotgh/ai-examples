import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

if __name__ == "__main__":

    # Load and preprocess data
    print("--- Data preprocessing ----")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("--x_train--")
    print(x_train)
    print("--x_test--")    
    print(x_test)

    # Reshape: (28, 28) -> (28, 28, 1)
    print("--- Reshape the inputs ----")
    x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
    x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0    

    # One-hot encode labels (optional for sparse_categorical_crossentropy)
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    # Build CNN model architecture
    print("----- Define the model architecture ------")
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 classes for MNIST
    ])
    print(model)

    # Compile model
    print("----- Compile the model ------")
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train model
    print("----- Train the model ------")
    model.fit(x_train, y_train_cat, epochs=5, batch_size=64, validation_split=0.1)
    print(model)

    # Evaluate test accutacy 
    print("----- Evaluate the model ------")
    test_loss, test_acc = model.evaluate(x_test, y_test_cat)
    print(f"Test accuracy: {test_acc:.4f}")

    print("----- Save the model weight ---")
    model.save('letter_classification_model.keras')
    print("Successfully saved this model!")    
