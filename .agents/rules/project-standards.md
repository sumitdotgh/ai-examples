# Repository Standards & Tooling Rules

These rules apply across the entire `ai-examples` workspace. Individual projects inherit these rules without needing per-project rule files.

## 1. Poetry & Dependency Management
- Every subproject must be an independent Poetry 2 project with a `pyproject.toml`.
- Always specify `package-mode = false` in `[tool.poetry]` since subprojects are application examples and not published libraries.
- Specify clean dependency version bounds:
  - `langchain (>=0.3.26,<0.4.0)`
  - `langchain-openai (>=0.3.28,<0.4.0)`
  - `langgraph (>=0.5.3,<0.6.0)`
  - `python-dotenv (>=1.1.1,<2.0.0)`
  - `pydantic (>=2.11.0,<3.0.0)`
- All subprojects must pass Pylint syntax validation: `poetry run pylint $(git ls-files '*.py')`.

## 2. Python Version Constraints
- **Agents, LangChain, LangGraph, FastAPI, MCP**: Use Python `>=3.12,<4.0`.
- **Keras 3, TensorFlow, Deep Learning**: Use Python `>=3.10,<3.12` (compatible with TensorFlow runtime wheels).

## 3. Environment Variables & Secret Handling
- Never hardcode secrets, tokens, or personal API keys.
- Always provide `.env.example` alongside `.env`.
- Ensure `.gitignore` ignores `.env`, `.env.*.local`, `*.log`, `.venv/`, `__pycache__/`, and `.DS_Store`.
- Load variables via `python-dotenv`:
  ```python
  from dotenv import load_dotenv
  load_dotenv()
  ```
- Pass tokens into LangChain using `pydantic.SecretStr`:
  ```python
  from pydantic import SecretStr
  secret = SecretStr(os.environ["GITHUB_TOKEN"])
  ```

## 4. Default Endpoints
- **LLM**: GitHub Models Inference (`https://models.github.ai/inference`), token in `GITHUB_TOKEN` or `GITHUB_MODELS_API_KEY`.
- **Embeddings**: Azure AI Inference (`https://models.inference.ai.azure.com`), model `text-embedding-3-large`, token in `EMBEDDING_MODEL_GITHUB_TOKEN`.

## 5. Tool Logging Pattern
Wrap tools with a logging decorator to track tool invocations:
```python
def tool_logger(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        logging.info(f"Calling tool {fn.__name__} with args={args} kwargs={kwargs}")
        return fn(*args, **kwargs)
    return wrapper
```
