## Basic RNN model

Basic RNN for the next Word Prediction

### Model Training

- Train the model using below command:

```sh
python model.py
```

- Model gets saved as `keras` format.

### Evaluate Model

- Test the model using below command:

```sh
python evaluate.py
```

### Visual Explaination

#### Training

```mermaid
flowchart TD
    A[Start Script] --> B[Prepare Text Corpus]
    B --> C[Tokenize Corpus with Keras Tokenizer]
    C --> D[Generate Input Sequences and Targets]
    D --> E[Pad Sequences to Fixed Length]
    E --> F[Convert Sequences to Numpy Arrays]

    F --> G[Build RNN Model with Embedding + SimpleRNN + Dense]
    G --> H[Compile Model with Adam and Crossentropy Loss]
    H --> I[Train Model on Input Data]
    I --> J[Save Trained Model as .keras]
```

#### Testing

```mermaid
flowchart TD
    A[Start Script] --> B[Load Trained RNN Model from File]
    B --> C[Recreate Corpus and Tokenizer]
    C --> D[Reconstruct Index-to-Word Mapping]
    D --> E[User Enters Input Phrase]
    E --> F[Tokenize and Pad the Input Phrase]
    F --> G[Predict Next Word Using Model]
    G --> H[Find Predicted Word from Index]
    H --> I[Display Predicted Word to User]
    I --> E
```