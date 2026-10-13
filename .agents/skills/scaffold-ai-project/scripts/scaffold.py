#!/usr/bin/env python3
"""
Repository-Level Project Scaffolder for ai-examples.

Scaffolds a new self-contained subproject adhering to all repository standards:
- Poetry 2 with package-mode = false
- Python versioning (3.12 for LLM/Agents/FastAPI, 3.11 for Keras/TensorFlow)
- .env.example, .gitignore, README.md, main.py
- Optional catalog registration (README.md, mkdocs.yml, docs/)
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

TEMPLATES = {
    "langchain-agent": {
        "python": ">=3.12,<4.0",
        "dependencies": [
            'langchain = ">=0.3.26,<0.4.0"',
            'langchain-openai = ">=0.3.28,<0.4.0"',
            'python-dotenv = ">=1.1.1,<2.0.0"',
            'pydantic = ">=2.11.0,<3.0.0"',
        ],
        "env_example": """GITHUB_TOKEN=your_github_token_here
GITHUB_MODELS_MODEL=openai/gpt-4.1-mini
GITHUB_MODELS_BASE_URL=https://models.github.ai/inference
""",
        "main_py": '''"""Main entry point for {project_name}."""
import os
import logging
from dotenv import load_dotenv
from pydantic import SecretStr, BaseModel
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent

load_dotenv()
logging.basicConfig(level=logging.INFO)

TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_MODELS_API_KEY", "")
ENDPOINT = os.environ.get("GITHUB_MODELS_BASE_URL", "https://models.github.ai/inference")
MODEL = os.environ.get("GITHUB_MODELS_MODEL", "openai/gpt-4.1-mini")

llm = ChatOpenAI(base_url=ENDPOINT, api_key=SecretStr(TOKEN), model=MODEL)

@tool
def sample_tool(query: str) -> str:
    """Processes a sample query."""
    return f"Processed query: {{query}}"

tools = [sample_tool]
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use your tools when needed."),
    ("placeholder", "{{messages}}"),
])

agent_executor = create_react_agent(model=llm, tools=tools, prompt=prompt)

def main():
    query = "Hello, what can you do?"
    response = agent_executor.invoke({{"messages": [("human", query)]}})
    print("Agent Response:", response["messages"][-1].content)

if __name__ == "__main__":
    main()
''',
    },
    "rag-system": {
        "python": ">=3.12,<4.0",
        "dependencies": [
            'langchain = ">=0.3.26,<0.4.0"',
            'langchain-openai = ">=0.3.28,<0.4.0"',
            'langchain-chroma = ">=0.2.4,<0.3.0"',
            'chromadb = ">=1.0.13,<2.0.0"',
            'python-dotenv = ">=1.1.1,<2.0.0"',
            'pydantic = ">=2.11.0,<3.0.0"',
        ],
        "env_example": """GITHUB_TOKEN=your_github_token_here
EMBEDDING_MODEL_GITHUB_TOKEN=your_embedding_token_here
AZURE_AI_INFERENCE_ENDPOINT=https://models.inference.ai.azure.com
""",
        "main_py": '''"""RAG system entry point for {project_name}."""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

load_dotenv()

def main():
    print("RAG Pipeline initialized for {project_name}")

if __name__ == "__main__":
    main()
''',
    },
    "fastapi-mcp": {
        "python": ">=3.12,<4.0",
        "dependencies": [
            'fastapi = ">=0.115.12,<0.116.0"',
            'uvicorn = ">=0.34.3,<0.35.0"',
            'fastapi-mcp = ">=0.3.0"',
            'pydantic = ">=2.11.0,<3.0.0"',
            'python-dotenv = ">=1.1.1,<2.0.0"',
        ],
        "env_example": """PORT=8000
HOST=0.0.0.0
""",
        "main_py": '''"""FastAPI MCP Service for {project_name}."""
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP
from pydantic import BaseModel

app = FastAPI(title="{project_name} MCP Service")

class Item(BaseModel):
    name: str
    description: str

@app.get("/items/{{item_id}}")
async def get_item(item_id: int):
    """Retrieve an item by ID."""
    return {{"item_id": item_id, "name": "Sample Item"}}

mcp = FastApiMCP(app, include_operations=["get_item"])
mcp.mount()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
''',
    },
    "keras-ml": {
        "python": ">=3.10,<3.12",
        "dependencies": [
            'keras = ">=3.0.0"',
            'tensorflow = ">=2.16.0"',
            'numpy = ">=1.24.0"',
            'pandas = ">=2.0.0"',
            'scikit-learn = ">=1.3.0"',
        ],
        "env_example": """# Add any dataset paths or flags here
""",
        "main_py": '''"""Keras Deep Learning model for {project_name}."""
import numpy as np
import tensorflow as tf
from keras import layers, models, saving

@saving.register_keras_serializable()
class CustomLayer(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w)

    def get_config(self):
        config = super().get_config()
        config.update({{"units": self.units}})
        return config

def main():
    print("TensorFlow / Keras 3 pipeline ready for {project_name}")

if __name__ == "__main__":
    main()
''',
    },
}

GITIGNORE_CONTENT = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.venv/
venv/
env/

# Environment
.env
.env.local
.env.*.local

# ML Models & data
*.keras
*.h5
*.pkl
data/
chroma_store/

# Logs & IDE
*.log
logs/
.DS_Store
.idea/
.vscode/
"""

README_TEMPLATE = """# {project_name}

{description}

## Prerequisites
- Python {python_version}
- Poetry 2.x

## Setup
```bash
poetry install
cp .env.example .env
# Fill in required environment variables in .env
```

## Run
```bash
poetry run python main.py
```
"""


def scaffold(name: str, archetype: str, description: str, author: str = "sumitdotgh"):
    target_dir = REPO_ROOT / name
    if target_dir.exists():
        print(f"Error: Directory '{name}' already exists at {target_dir}")
        sys.exit(1)

    template = TEMPLATES[archetype]
    target_dir.mkdir(parents=True)

    # pyproject.toml
    deps_str = "\n".join(f'    "{dep.split()[0]} ({dep.split()[2]})",' for dep in template["dependencies"])
    pyproject_content = f"""[project]
name = "{name}"
version = "0.1.0"
description = "{description}"
authors = [
    {{name = "{author}"}}
]
readme = "README.md"
requires-python = "{template['python']}"
dependencies = [
{deps_str}
]

[tool.poetry]
package-mode = false

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"
"""
    (target_dir / "pyproject.toml").write_text(pyproject_content)
    (target_dir / ".env.example").write_text(template["env_example"])
    (target_dir / ".gitignore").write_text(GITIGNORE_CONTENT)
    (target_dir / "README.md").write_text(README_TEMPLATE.format(
        project_name=name,
        description=description,
        python_version=template["python"]
    ))
    (target_dir / "main.py").write_text(template["main_py"].format(project_name=name))

    print(f"Successfully scaffolded '{name}' under '{target_dir}' using archetype '{archetype}'!")


def main():
    parser = argparse.ArgumentParser(description="Scaffold a new project in ai-examples")
    parser.add_argument("name", help="Name of the new subproject (e.g., basic-vision-agent)")
    parser.add_argument(
        "--archetype",
        choices=list(TEMPLATES.keys()),
        default="langchain-agent",
        help="Project archetype pattern"
    )
    parser.add_argument("--description", default="AI example project", help="Short description")
    parser.add_argument("--author", default="sumitdotgh", help="Author name")

    args = parser.parse_args()
    scaffold(args.name, args.archetype, args.description, args.author)


if __name__ == "__main__":
    main()
