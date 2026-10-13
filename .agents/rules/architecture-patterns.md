# Core Architecture Patterns

This rule defines the repository's 5 primary design archetypes. All projects created in or added to `ai-examples` must conform to these blueprints.

---

## 1. Single ReAct Agent (`basic-agent*`)
- **Framework**: LangChain, `ChatOpenAI`, GitHub Models (`openai/gpt-4.1` or `openai/gpt-4.1-mini`).
- **Tools**: Defined via `@tool` with typing and docstrings, or `StructuredTool.from_function` with explicit Pydantic `BaseModel` schema.
- **Agent Construction**: Use `create_react_agent` or `create_openai_tools_agent`. Avoid deprecated `initialize_agent`.
- **System Prompts**: Formatted using `ChatPromptTemplate.from_messages` with `MessagesPlaceholder("chat_history", optional=True)` and `MessagesPlaceholder("agent_scratchpad")`.

---

## 2. Multi-Agent Systems (`basic-multi-agent-system`)
- **Framework**: LangGraph (`StateGraph`), LangChain Core Runnables.
- **Pattern**:
  - `state.py`: Defines shared Pydantic / TypedDict state schema across agent steps.
  - `agents/`: Independent agent functions (e.g., `controller_agent`, `planner_agent`, `action_agent`).
  - Graph assembly: Wrap agent functions with `RunnableLambda`, add nodes with `builder.add_node()`, establish flow via `builder.add_edge()`, and compile.

---

## 3. RAG Pipelines (`basic-rag`)
- **Framework**: LangChain, ChromaDB (`langchain-chroma`), Azure AI Inference or GitHub embeddings (`text-embedding-3-large`).
- **Separation of Concerns**:
  - `load_data.py`: Loads raw documents, chunks via `RecursiveCharacterTextSplitter`, embeds, and persists to `./chroma_store`.
  - `main.py`: Loads persisted Chroma store, initializes LLM, builds `RetrievalQA` with custom prompt template, and runs user queries.

---

## 4. FastAPI & MCP Servers (`fastapi-mcp-api`, `test-mcp-client`)
- **Framework**: FastAPI, `fastapi-mcp`, `httpx`, Pydantic v2.
- **Structure**:
  - `routes/`: Modular APIRouter endpoints.
  - `models.py`: Pydantic request and response schemas.
  - `main.py`: Instantiates `FastApiMCP(app, include_operations=[...])` and executes `mcp.mount()`.
  - Client side: Async client leveraging `httpx.AsyncClient` to consume MCP endpoints.

---

## 5. Keras 3 & Deep Learning Models (`basic-cnn`, `tiny-moe-based-model`, `basic-peft`)
- **Framework**: TensorFlow, Keras 3 (`keras>=3.0.0`), NumPy, Scikit-learn.
- **Layer Design**: Custom layers must inherit from `keras.layers.Layer` or `layers.Layer` and include `@saving.register_keras_serializable()`. Must implement `build(self, input_shape)`, `call(self, inputs)`, and `get_config(self)`.
- **Persistence**: Save trained models to modern `.keras` zip format (e.g., `model.save("model.keras")`).
- **Serving / Evaluation**: Provide standalone `evaluate.py` or `serve.py` for inference testing.
