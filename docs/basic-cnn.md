## Basic CNN model

### Prerequisite

- For downloading the MNIST dataset under tensorflow.keras

```sh
open /Applications/Python\ 3.x/Install\ Certificates.command
```

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
    A[Load MNIST Dataset] --> B[Preprocess Data]
    B --> C[Encode Labels]
    C --> D[Define CNN Model]
    D --> E[Compile Model]
    E --> F[Train Model]
    F --> G[Evaluate on Test Set]
    G --> H[Save Trained Model]    
```

#### Testing

```mermaid
flowchart TD
    A[Load Trained Model] --> B[Load Test Image]
    B --> C[Preprocess Image]
    C --> D[Make Prediction]
    D --> E[Map Class Index to Label]
    E --> F[Print Predicted Letter]
```