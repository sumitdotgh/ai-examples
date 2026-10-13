# AI Project Setup Guide

## Overview
This comprehensive guide provides step-by-step instructions for setting up new AI/ML projects in this repository, covering all project types: Machine Learning, LangChain Agents, RAG systems, and FastAPI services.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Project Types](#project-types)
3. [Standard Project Structure](#standard-project-structure)
4. [Configuration Files](#configuration-files)
5. [Project Type Specific Setup](#project-type-specific-setup)
6. [Common Patterns](#common-patterns)
7. [Best Practices](#best-practices)
8. [Quick Start Commands](#quick-start-commands)

---

## Prerequisites

### Required Software
- **Python**: 3.10+ (for ML projects) or 3.12+ (for LangChain/Agent projects)
- **Poetry**: Package and dependency management
- **Git**: Version control
- **Docker**: For containerized deployments (optional)

### Installation
```bash
# Install Poetry (macOS)
curl -sSL https://install.python-poetry.org | python3 -

# Verify installation
poetry --version
python3 --version
git --version
```

---

## Project Types

This repository contains four main project types:

| Project Type | Technologies | Example Projects | Python Version |
|--------------|--------------|------------------|----------------|
| **Machine Learning** | TensorFlow, Keras, scikit-learn | basic-dnn, basic-cnn, basic-rnn, basic-peft | 3.10+ |
| **LangChain Agent** | LangChain, OpenAI, LangGraph | basic-agent, basic-multi-agent-system | 3.12+ |
| **RAG System** | LangChain, ChromaDB, Embeddings | basic-rag | 3.12+ |
| **FastAPI Service** | FastAPI, Uvicorn, MCP | fastapi-mcp-api | 3.12+ |

---

## Standard Project Structure

### Recommended Directory Layout

```
project-name/
├── pyproject.toml          # Project configuration and dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore patterns
├── .env                    # Environment variables (not committed)
├── .env.example            # Environment variable template (committed)
├── main.py                 # Main entry point
├── model.py                # Model definition (for ML projects)
├── __init__.py             # Package initialization (if needed)
├── data/                   # Data directory (gitignored)
│   └── .gitkeep
├── tests/                  # Test files (optional)
│   └── test_main.py
└── docs/                   # Additional documentation (optional)
    └── architecture.md
```

### For Multi-Module Projects

```
project-name/
├── pyproject.toml
├── README.md
├── .gitignore
├── .env
├── main.py
├── state.py                # Shared state/configuration
├── agents/                 # Agent modules
│   ├── __init__.py
│   ├── research_agent.py
│   └── writer_agent.py
├── models/                 # Data models
│   └── schemas.py
├── routes/                 # API routes (FastAPI)
│   └── api.py
├── repositories/           # Data access layer
│   └── data_repo.py
└── utils/                  # Utility functions
    └── helpers.py
```

---

## Configuration Files

### 1. pyproject.toml Templates

#### Machine Learning Projects (TensorFlow/Keras)

```toml
[project]
name = "project-name"
version = "0.1.0"
description = "Brief description of the ML model and its purpose"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "keras>=3.0.0",
    "tensorflow>=2.16.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "numpy>=1.24.0",
]

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "project-name"
version = "0.1.0"
description = "Brief description"
authors = []
package-mode = false
```

#### LangChain/Agent Projects

```toml
[project]
name = "project-name"
version = "0.1.0"
description = "AI agent with LangChain"
authors = [
    {name = "Your Name"}
]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.12,<4.0"
dependencies = [
    "langchain (>=0.3.25,<0.4.0)",
    "langchain-openai (>=0.3.20,<0.4.0)",
    "python-dotenv (>=1.1.0,<2.0.0)",
    "langchain-community (>=0.3.24,<0.4.0)",
    "langgraph (>=0.4.8,<0.5.0)",
    "pylint (>=3.3.7,<4.0.0)"
]

[tool.poetry]
package-mode = false

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"
```

#### RAG System Projects

```toml
[project]
name = "project-name"
version = "0.1.0"
description = "RAG system with vector search"
authors = [
    {name = "Your Name"}
]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.12,<4.0"
dependencies = [
    "langgraph (>=0.4.8,<0.5.0)",
    "chromadb (>=1.0.13,<2.0.0)",
    "langchain (>=0.3.26,<0.4.0)",
    "azure-ai-inference (>=1.0.0b9,<2.0.0)",
    "langchain-community (>=0.3.26,<0.4.0)",
    "langchain-chroma (>=0.2.4,<0.3.0)",
    "langchain-openai (>=0.3.24,<0.4.0)"
]

[tool.poetry]
package-mode = false

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"
```

#### FastAPI Service Projects

```toml
[project]
name = "project-name"
version = "0.1.0"
description = "FastAPI service"
authors = [
    {name = "Your Name"}
]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.12,<4.0"
dependencies = [
    "fastapi (>=0.115.12,<0.116.0)",
    "uvicorn (>=0.34.3,<0.35.0)",
    "pylint (>=3.3.7,<4.0.0)"
]

[tool.poetry]
package-mode = false

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"
```

### 2. Environment Variables (.env)

Create a `.env` file for sensitive configuration (never commit this):

```bash
# API Keys and Tokens
OPENAI_API_KEY=your_key_here
GITHUB_TOKEN=your_github_token_here
EMBEDDING_MODEL_GITHUB_TOKEN=your_embedding_token_here

# Model Configuration
MODEL_NAME=openai/gpt-4.1
ENDPOINT=https://models.github.ai/inference

# LangSmith (for tracing, optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=project-name

# Environment
PROJECT_ENV=development
DEBUG=true
```

Create a `.env.example` (commit this as template):

```bash
# Copy to .env and fill in actual values
OPENAI_API_KEY=
GITHUB_TOKEN=
EMBEDDING_MODEL_GITHUB_TOKEN=

MODEL_NAME=openai/gpt-4.1
ENDPOINT=https://models.github.ai/inference

PROJECT_ENV=development
DEBUG=true
```

### 3. .gitignore Template

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/
.eggs/

# Virtual Environments
.venv/
venv/
ENV/
env/

# Environment Variables
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*.swn
.DS_Store

# ML Models and Data
*.h5
*.keras
*.pkl
*.joblib
*.pth
*.ckpt
data/
*.csv
*.json
!.env.example

# Vector Stores
chroma_store/
*.db
*.index

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db
*.bak

# Jupyter Notebooks
.ipynb_checkpoints/
*.ipynb

# Docker
docker-compose.override.yml
```

### 4. README.md Template

```markdown
# Project Name

Brief description of what this project does.

## Overview
Detailed explanation of the project, its purpose, and key features.

## Prerequisites
- Python 3.10+ (or 3.12+ for LangChain projects)
- Poetry
- [Any other specific requirements]

## Installation

1. Clone the repository
```bash
cd ai-examples/project-name
```

2. Install dependencies
```bash
poetry install
```

3. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your actual values
```

## Usage

```bash
poetry run python main.py
```

## Project Structure
```
project-name/
├── main.py           # Main entry point
├── model.py          # [Description]
└── ...
```

## Configuration
Explain key configuration options and environment variables.

## Examples
Provide usage examples and expected output.

## License
MIT
```

---

## Project Type Specific Setup

### Machine Learning Projects

#### Code Structure Pattern

```python
"""
[Model Name] - [Brief Description]
==========================================

Architecture:
- Input: [dimensions and format]
- Output: [dimensions and format]
- Layers: [architecture overview]
"""

import numpy as np
import keras
from keras import layers, models
from keras.utils import to_categorical
import tensorflow as tf

# Enable GPU logging (optional)
tf.debugging.set_log_device_placement(True)

def print_device_info():
    """Print GPU/CPU device information."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs detected: {[gpu.name for gpu in gpus]}")
    else:
        print("No GPU detected. Running on CPU.")

def load_and_preprocess_data():
    """
    Load and preprocess dataset.
    
    Returns:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
    """
    # Load data (example: MNIST)
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize to [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    
    # Reshape for CNN (if needed)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return X_train, X_test, y_train, y_test

def build_model(input_shape, num_classes=10):
    """
    Build the neural network model.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    # For multi-GPU support
    strategy = tf.distribute.MirroredStrategy()
    
    with strategy.scope():
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        
        # Add layers
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.2))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(num_classes, activation='softmax'))
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=128):
    """Train the model with callbacks."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Model Training")
    print("=" * 60)
    
    print_device_info()
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Build model
    model = build_model(input_shape=X_train.shape[1:])
    model.summary()
    
    # Train
    print("\nTraining...")
    train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluate
    print("\nEvaluating...")
    evaluate_model(model, X_test, y_test)
    
    # Save
    model.save("model.keras")
    print("\nModel saved to 'model.keras'")

if __name__ == "__main__":
    main()
```

### LangChain Agent Projects

#### Code Structure Pattern

```python
"""
[Agent Name] - [Brief Description]
"""

import os
from dotenv import load_dotenv
from pydantic import SecretStr

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()

# Configuration
TOKEN = os.environ["GITHUB_TOKEN"]
ENDPOINT = "https://models.github.ai/inference"
MODEL = "openai/gpt-4.1"

# Initialize LLM
llm = ChatOpenAI(
    base_url=ENDPOINT,
    api_key=SecretStr(TOKEN),
    model=MODEL
)

# Define tools
@tool
def tool_name(param: str) -> str:
    """Clear description of what this tool does."""
    # Implementation
    return f"Result for {param}"

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a [role] assistant.
        
        Your responsibilities:
        1. [Task 1]
        2. [Task 2]
        
        Guidelines:
        - [Guideline 1]
        - [Guideline 2]
        """,
    ),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# Setup agent
tools = [tool_name]
agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

def main():
    """Main execution."""
    query = "Your query here"
    result = agent_executor.invoke({"input": query})
    print(result["output"])

if __name__ == "__main__":
    main()
```

### RAG System Projects

#### Code Structure Pattern

```python
"""
RAG System - [Description]
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential

load_dotenv()

# Configuration
GITHUB_INFERENCE_ENDPOINT = "https://models.github.ai/inference"
GPT_MODEL_NAME = "openai/gpt-4.1"
GPT_MODEL_TOKEN = os.environ["GPT_4_1_MODEL_GITHUB_TOKEN"]

AZURE_AI_INFERENCE_ENDPOINT = "https://models.inference.ai.azure.com"
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
EMBEDDING_MODEL_TOKEN = os.environ["EMBEDDING_MODEL_GITHUB_TOKEN"]

DATA_DIR = "./data"
PERSISTED_DIR = "./chroma_store"
COLLECTION_NAME = "documents"

# Custom Embeddings
class AzureInferenceOpenAIEmbeddings(Embeddings):
    def __init__(self, endpoint: str, token: str, model: str):
        self.client = EmbeddingsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(token)
        )
        self.model = model
    
    def embed_documents(self, texts):
        response = self.client.embed(input=texts, model=self.model)
        return [item["embedding"] for item in response["data"]]
    
    def embed_query(self, text):
        response = self.client.embed(input=[text], model=self.model)
        return response["data"][0]["embedding"]

def load_documents(directory: str):
    """Load documents from directory."""
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith('.txt'):
            with open(filepath, 'r') as f:
                content = f.read()
                documents.append(Document(
                    page_content=content,
                    metadata={"source": filename}
                ))
    return documents

def setup_rag_system(rebuild_index=False):
    """Setup RAG system with vector store."""
    
    # Initialize embeddings
    embedding_model = AzureInferenceOpenAIEmbeddings(
        endpoint=AZURE_AI_INFERENCE_ENDPOINT,
        token=EMBEDDING_MODEL_TOKEN,
        model=EMBEDDING_MODEL_NAME
    )
    
    # Load or create vector store
    if rebuild_index or not os.path.exists(PERSISTED_DIR):
        documents = load_documents(DATA_DIR)
        
        # Chunk documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunked_docs = text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=chunked_docs,
            embedding=embedding_model,
            persist_directory=PERSISTED_DIR,
            collection_name=COLLECTION_NAME
        )
    else:
        vectorstore = Chroma(
            persist_directory=PERSISTED_DIR,
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model
        )
    
    # Initialize LLM
    llm = ChatOpenAI(
        base_url=GITHUB_INFERENCE_ENDPOINT,
        api_key=GPT_MODEL_TOKEN,
        model=GPT_MODEL_NAME
    )
    
    # Create prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Answer based on the context below.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return qa_chain

def main():
    """Main execution."""
    qa_chain = setup_rag_system(rebuild_index=False)
    
    question = "Your question here"
    result = qa_chain.invoke({"query": question})
    
    print(f"Q: {question}")
    print(f"A: {result['result']}")
    print(f"Sources: {[doc.metadata['source'] for doc in result['source_documents']]}")

if __name__ == "__main__":
    main()
```

### FastAPI Service Projects

#### Code Structure Pattern

```python
"""
FastAPI Service - [Description]
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="API Name",
    description="API Description",
    version="1.0.0"
)

# Models
class RequestModel(BaseModel):
    field1: str
    field2: int

class ResponseModel(BaseModel):
    result: str
    status: str

# Routes
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "API Name"}

@app.post("/endpoint", response_model=ResponseModel)
async def endpoint(request: RequestModel):
    """Endpoint description."""
    try:
        # Process request
        result = f"Processed {request.field1}"
        return ResponseModel(result=result, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Common Patterns

### 1. Loading Environment Variables

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Required variables
REQUIRED_VARS = ["GITHUB_TOKEN", "MODEL_NAME"]

for var in REQUIRED_VARS:
    if var not in os.environ:
        raise EnvironmentError(f"Missing required environment variable: {var}")

# Access variables
token = os.environ["GITHUB_TOKEN"]
model = os.environ.get("MODEL_NAME", "default-model")
```

### 2. Logging Setup

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### 3. Error Handling

```python
def safe_execution(func):
    """Decorator for safe function execution."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

@safe_execution
def risky_function():
    # Your code here
    pass
```

### 4. Configuration Management

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Application configuration."""
    model_name: str
    endpoint: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment."""
        return cls(
            model_name=os.environ["MODEL_NAME"],
            endpoint=os.environ["ENDPOINT"],
            temperature=float(os.environ.get("TEMPERATURE", "0.7")),
            max_tokens=int(os.environ["MAX_TOKENS"]) if "MAX_TOKENS" in os.environ else None
        )

config = Config.from_env()
```

---

## Best Practices

### 1. Dependency Management
- Use specific version ranges: `>=0.3.25,<0.4.0`
- Pin major versions to avoid breaking changes
- Keep dependencies minimal and necessary
- Use `poetry.lock` for reproducible builds
- Regularly update dependencies for security

### 2. Code Organization
- **Separation of Concerns**: One responsibility per file/module
- **Clear Naming**: Descriptive function and variable names
- **Type Hints**: Use type annotations for all function parameters
- **Docstrings**: Document all functions with purpose, args, and returns
- **Comments**: Explain complex logic, not obvious code

### 3. Documentation
- **README**: Clear setup and usage instructions
- **Code Comments**: Explain why, not what
- **Examples**: Provide working examples
- **Architecture Diagrams**: For complex systems
- **API Documentation**: For services

### 4. Version Control
- **Atomic Commits**: One logical change per commit
- **Clear Messages**: Descriptive commit messages
- **Branch Strategy**: Use feature branches
- **Pull Requests**: Review before merging
- **.gitignore**: Exclude sensitive and generated files

### 5. Security
- **Never commit secrets**: Use environment variables
- **Validate inputs**: Sanitize all user inputs
- **Error messages**: Don't expose sensitive information
- **Dependencies**: Keep updated for security patches
- **API keys**: Rotate regularly

### 6. Testing
- **Unit Tests**: Test individual functions
- **Integration Tests**: Test component interactions
- **Edge Cases**: Test boundary conditions
- **Mock External Services**: Don't depend on external APIs in tests

### 7. Performance
- **Profile First**: Measure before optimizing
- **Cache Results**: For expensive operations
- **Batch Processing**: When possible
- **Async Operations**: For I/O-bound tasks
- **GPU Utilization**: For ML workloads

---

## Quick Start Commands

### Initialize New Project

```bash
# 1. Create project directory
mkdir project-name
cd project-name

# 2. Initialize git
git init

# 3. Create pyproject.toml (copy from templates above)
touch pyproject.toml

# 4. Install dependencies
poetry install

# 5. Create environment file
touch .env
touch .env.example

# 6. Create .gitignore
touch .gitignore

# 7. Create main script
touch main.py

# 8. Create README
touch README.md

# 9. Create data directory
mkdir -p data
touch data/.gitkeep
```

### Common Poetry Commands

```bash
# Install dependencies
poetry install

# Add new dependency
poetry add package-name

# Add development dependency
poetry add --group dev package-name

# Remove dependency
poetry remove package-name

# Update dependencies
poetry update

# Update specific package
poetry update package-name

# Show installed packages
poetry show

# Show dependency tree
poetry show --tree

# Run Python script
poetry run python main.py

# Activate virtual environment
poetry shell

# Deactivate virtual environment
exit

# Export requirements.txt
poetry export -f requirements.txt --output requirements.txt
```

### Running Projects

```bash
# Machine Learning
cd basic-dnn
poetry install
poetry run python model.py

# LangChain Agent
cd basic-agent
cp .env.example .env  # Edit with your keys
poetry install
poetry run python main.py

# RAG System
cd basic-rag
cp .env.example .env  # Edit with your keys
poetry install
poetry run python load_data.py  # First time only
poetry run python main.py

# FastAPI Service
cd fastapi-mcp-api
poetry install
poetry run uvicorn main:app --reload
# Access at http://localhost:8000
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `poetry: command not found` | Install Poetry: `curl -sSL https://install.python-poetry.org \| python3 -` |
| Python version mismatch | Use pyenv to install correct version: `pyenv install 3.12` |
| Dependency conflicts | Clear cache: `poetry cache clear pypi --all && poetry install` |
| Import errors | Ensure virtual env is activated: `poetry shell` |
| Environment variables not loading | Check `.env` file exists and `load_dotenv()` is called |
| GPU not detected | Check CUDA installation and TensorFlow GPU version |
| Out of memory (GPU) | Reduce batch size or model complexity |
| Slow training | Enable GPU or reduce model size |

### Getting Help

1. Check project-specific README
2. Review example projects in the repository
3. Consult official documentation:
   - [TensorFlow](https://www.tensorflow.org/)
   - [LangChain](https://python.langchain.com/)
   - [FastAPI](https://fastapi.tiangolo.com/)
4. Check error logs in `app.log`

---

## Additional Resources

### Documentation Links
- **Poetry**: https://python-poetry.org/docs/
- **TensorFlow**: https://www.tensorflow.org/guide
- **Keras**: https://keras.io/guides/
- **LangChain**: https://python.langchain.com/docs/
- **FastAPI**: https://fastapi.tiangolo.com/
- **ChromaDB**: https://docs.trychroma.com/

### Example Projects in This Repository
- **basic-dnn**: Simple deep neural network
- **basic-cnn**: Convolutional neural network for images
- **basic-rnn**: Recurrent neural network for sequences
- **basic-agent**: LangChain agent with tools
- **basic-rag**: RAG system with vector search
- **fastapi-mcp-api**: FastAPI service with MCP
- **basic-multi-agent-system**: Multi-agent coordination with LangGraph

---

## Checklist for New Projects

- [ ] Create project directory
- [ ] Initialize git repository
- [ ] Create and configure `pyproject.toml`
- [ ] Create `.env.example` template
- [ ] Create `.gitignore` from template
- [ ] Install dependencies with Poetry
- [ ] Create main script with proper structure
- [ ] Add comprehensive docstrings
- [ ] Create README with setup instructions
- [ ] Test basic functionality
- [ ] Add to repository documentation (if needed)
- [ ] Create initial git commit

---

**Last Updated**: March 7, 2026
