---
name: fastapi-mcp-service
description: >-
  Guides developing Model Context Protocol (MCP) servers and clients using FastAPI and fastapi-mcp. Use this skill whenever creating, updating, or debugging MCP tools, endpoints, or client adapters.
---

# FastAPI MCP Service Skill

Use this skill to implement Model Context Protocol (MCP) APIs and test clients following the repo conventions.

## Server Checklist

1. **APIRouter Organization**:
   - Organize domain endpoints into `routes/<domain>.py`.
   - Specify explicit `operation_id` on route decorators (e.g., `operation_id="get_weather_info_by_city"`).

2. **Mount FastApiMCP**:
   - In `main.py`, create `FastApiMCP(app, include_operations=[...])` listing the exposed operations.
   - Mount the protocol using `mcp.mount()`.

3. **Data Contracts**:
   - Define all request payloads and response models in `models.py` using Pydantic v2.

4. **Client Testing**:
   - Create client wrappers using async `httpx.AsyncClient` matching the pattern in `test-mcp-client`.

Review [references/patterns.md](./references/patterns.md) for code templates.
