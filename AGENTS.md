# ai-examples Repository Guidelines & Architecture Patterns

This file defines the mandatory engineering standards, architectural patterns, and project conventions for all AI coding agents working in the `ai-examples` repository.

Refer to [GEMINI.md](file:///Users/sumitghosh/source/github/ai-examples/GEMINI.md) for the primary specification.

## Quick Summary for Agents

1. **Package Manager**: Use **Poetry 2.x** with `package-mode = false` in `pyproject.toml`.
2. **Python Versions**: Python 3.12+ for LangChain, LangGraph, Agents, FastAPI; Python 3.10-3.11 for Keras/TensorFlow.
3. **Secrets & Environment**: Never commit `.env`. Always include `.env.example`.
4. **LLM Endpoint**: Default to GitHub Models Inference (`https://models.github.ai/inference`, `GITHUB_TOKEN`).
5. **Architectural Archetypes**:
   - `basic-agent`: LangChain `create_openai_tools_agent` or `create_react_agent`.
   - `basic-multi-agent-system`: LangGraph `StateGraph` with explicit nodes and state.
   - `basic-rag`: ChromaDB, Azure AI Inference embeddings, `RetrievalQA`.
   - `fastapi-mcp-*`: FastAPI with `FastApiMCP(app, ...).mount()`.
   - `basic-*` (ML): Keras 3 with `@saving.register_keras_serializable()`, `.keras` saves.
6. **Documentation Sync**: When adding a new project, always update root `README.md`, `mkdocs.yml`, `docs/<project>.md`, and `.github/workflows/ci.yml`.
7. **Workspace Skills**: Located in `.agents/skills/`. Use `scaffold-ai-project`, `fastapi-mcp-service`, and `keras-deep-learning`.
