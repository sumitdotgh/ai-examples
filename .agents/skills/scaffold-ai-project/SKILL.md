---
name: scaffold-ai-project
description: >-
  Scaffolds a new project in the ai-examples repository adhering to standard repository conventions. Use this skill whenever the user asks to create, initialize, or scaffold a new subproject, agent, multi-agent workflow, RAG pipeline, FastAPI MCP service, or deep learning model.
---

# Scaffold AI Project Skill

This skill automates the creation of a new subproject in `ai-examples` matching the repository's strict standards.

## Supported Archetypes

1. **`langchain-agent`**: Single ReAct agent with tools, structured schemas, and GitHub Models inference (matches `basic-agent`, `basic-agent-langgraph`).
2. **`rag-system`**: Retrieval-Augmented Generation with ChromaDB and Azure AI embeddings (matches `basic-rag`).
3. **`fastapi-mcp`**: FastAPI application exposing Model Context Protocol endpoints via `FastApiMCP` (matches `fastapi-mcp-api`).
4. **`keras-ml`**: Deep learning pipeline with Keras 3, TensorFlow, serializable custom layers, and `.keras` model saving (matches `basic-cnn`, `basic-peft`).

## Automated Scaffolding Procedure

### 1. Run Scaffolding Helper
Execute the CLI scaffolder from the repository root:

```bash
python3 .agents/skills/scaffold-ai-project/scripts/scaffold.py <project-name> --archetype <archetype> --description "<description>"
```

Example:
```bash
python3 .agents/skills/scaffold-ai-project/scripts/scaffold.py basic-vision-agent --archetype langchain-agent --description "Vision agent with multimodal tools"
```

This generates:
- `<project-name>/pyproject.toml` (Poetry 2, `package-mode = false`, exact Python version, and dependency ranges)
- `<project-name>/.env.example`
- `<project-name>/.gitignore`
- `<project-name>/README.md`
- `<project-name>/main.py`

### 2. Install Dependencies
Change into the newly created folder and verify the environment:
```bash
cd <project-name>
poetry install
cp .env.example .env
```

### 3. Synchronize Repository Catalog
After scaffolding the code, update the repository documentation:
1. Add an entry to the table in root `README.md`.
2. Add an entry to `mkdocs.yml` under `nav:`.
3. Create `docs/<project-name>.md` with an overview and Mermaid architecture diagram.
4. If suitable for automated CI linting, add `<project-name>` to `.github/workflows/ci.yml`.
