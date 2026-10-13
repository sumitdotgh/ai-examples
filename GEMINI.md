# ai-examples Repository Guidelines & Architecture Patterns

This file defines the engineering standards, architectural patterns, and project conventions for all AI assistants and developers contributing to `ai-examples`.

---

## 1. Project Philosophy & Structure

`ai-examples` is a curated collection of focused, clean, reproducible AI and machine learning projects. Each project is an independent, self-contained subproject in the repository root.

### Directory Naming Conventions
- Basic educational examples: `basic-<concept>` (e.g., `basic-agent`, `basic-rag`, `basic-cnn`, `basic-peft`)
- Architecture/mechanism explorations: `tiny-<concept>` (e.g., `tiny-gpt-model`, `tiny-moe-based-model`, `tiny-nested-learning`)
- Multi-agent / workflow systems: `multi-agents-workflow-<framework>` or `basic-multi-agent-system`
- Protocols & APIs: `<framework>-<protocol>-api` or `<protocol>-client` (e.g., `fastapi-mcp-api`, `test-mcp-client`)

### Subproject Structure Requirements
Every subproject in this repository MUST have:
1. `pyproject.toml` - Configured with Poetry 2 standards (`package-mode = false`).
2. `.env.example` - Template showing required environment variables with dummy values.
3. `.gitignore` - Ignoring `.venv/`, `.env`, `*.log`, `__pycache__/`, `*.keras`, and data files.
4. `README.md` - Clean documentation explaining purpose, prerequisites, setup, and run commands.
5. `main.py` (or `model.py`) - Main entry point with clear output logging.

---

## 2. Python & Dependency Management Standards

### Poetry Configuration
- **Package Manager**: Always use **Poetry** (version 2.x). Never use raw `pip` or setup `requirements.txt` unless required for external Docker builds (e.g. `basic-vllm`).
- **Package Mode**: Always specify `package-mode = false` in `[tool.poetry]` since subprojects are application examples, not published PyPI packages.
- **Python Versioning**:
  - **LangChain / LangGraph / Agents / FastAPI**: Python `requires-python = ">=3.12,<4.0"`.
  - **TensorFlow / Keras 3 / ML**: Python `requires-python = ">=3.10,<3.12"` (or compatible with TensorFlow wheels).

#### Standard `pyproject.toml` Template
```toml
[project]
name = "project-name"
version = "0.1.0"
description = "Brief, descriptive summary"
authors = [
    {name = "Your Name", email = "your-email@example.com"}
]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.12,<4.0"
dependencies = [
    "langchain (>=0.3.26,<0.4.0)",
    "langchain-openai (>=0.3.28,<0.4.0)",
    "python-dotenv (>=1.1.1,<2.0.0)",
    "pydantic (>=2.11.0,<3.0.0)",
]

[tool.poetry]
package-mode = false

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"
```

---

## 3. Environment Variables & Inference Provider Standards

### Security & Secret Handling
- **Never hardcode secrets, API keys, or personal tokens.**
- Always load environment variables using `python-dotenv`:
  ```python
  from dotenv import load_dotenv
  load_dotenv()
  ```
- Wrap sensitive API keys in `pydantic.SecretStr` when passing to LangChain components:
  ```python
  from pydantic import SecretStr
  secret = SecretStr(os.environ["GITHUB_TOKEN"])
  ```

### Default Model Endpoints (GitHub Models & Azure AI)
This repository prioritizes **GitHub Models Inference** and **Azure AI Inference**:
- **LLM Endpoint**: `https://models.github.ai/inference`
- **Default LLM Models**: `openai/gpt-4.1`, `openai/gpt-4.1-mini`, `openai/gpt-4.1-nano`
- **Token Variable**: `GITHUB_TOKEN` or `GITHUB_MODELS_API_KEY`
- **LangChain Client Setup**:
  ```python
  from langchain_openai import ChatOpenAI
  from pydantic import SecretStr

  token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_MODELS_API_KEY")
  llm = ChatOpenAI(
      base_url="https://models.github.ai/inference",
      api_key=SecretStr(token),
      model=os.environ.get("GITHUB_MODELS_MODEL", "openai/gpt-4.1-mini"),
  )
  ```
- **Azure Embeddings Endpoint**: `https://models.inference.ai.azure.com`
- **Default Embeddings Model**: `text-embedding-3-large`
- **Embedding Token Variable**: `EMBEDDING_MODEL_GITHUB_TOKEN`

---

## 4. Architectural Patterns

### A. LangChain & Tool Usage
- Use the `@tool` decorator or `StructuredTool.from_function` with Pydantic schemas.
- Include informative docstrings in every tool—the LLM reads docstrings to choose tools.
- Implement tool invocation logging via a decorator (see `basic-agent-langgraph` pattern).
- Avoid legacy `initialize_agent`; use `create_react_agent` or `create_openai_tools_agent`.

### B. Multi-Agent Systems
- Follow the patterns in `basic-multi-agent-system` using LangGraph `StateGraph`.
- Define shared state models using Pydantic or TypedDict.
- Use explicit nodes (`RunnableLambda`) and conditional/direct edges (`add_edge`).

### C. RAG Systems
- Split text with `RecursiveCharacterTextSplitter`.
- Use ChromaDB (`langchain-chroma`) for vector storage with persistent directory `./chroma_store`.
- Wrap embedding clients in a clean `Embeddings` class.
- Separate document ingestion (`load_data.py`) from query pipeline (`main.py`).

### D. FastAPI & Model Context Protocol (MCP)
- Use `FastAPI` with modular APIRouters in `routes/`.
- Use `fastapi-mcp` to expose operations as an MCP server via `FastApiMCP.mount()`.
- Define data contracts using Pydantic models in `models.py`.

### E. Keras 3 / Deep Learning
- Annotate custom layers with `@saving.register_keras_serializable()`.
- Implement `build()`, `call()`, `get_config()`, and `from_config()` for custom layers.
- Save models in modern `.keras` zip format (e.g., `model.save("model.keras")`).
- Use callbacks: `EarlyStopping(restore_best_weights=True)` and `ReduceLROnPlateau`.

---

## 5. Documentation & Catalog Synchronization Rule

Whenever a new subproject is created, modified, or renamed:
1. **Root README.md**: Add an entry to the `List of examples` table with an emoji and concise description.
2. **MkDocs Nav**: Add the project to `mkdocs.yml` under `nav:`.
3. **Docs Page**: Create `docs/<project-name>.md` with an overview, architecture (Mermaid diagram), setup, and usage.
4. **CI Workflow**: If applicable, add the project to the test matrix in `.github/workflows/ci.yml`.

---

## 6. Available Repository Skills

The repository contains specialized workspace skills in `.agents/skills/`:
- `scaffold-ai-project`: Automates creating a new subproject with full boilerplate.
- `fastapi-mcp-service`: Guides building FastAPI MCP servers and clients.
- `keras-deep-learning`: Guides developing Keras 3 neural networks.
