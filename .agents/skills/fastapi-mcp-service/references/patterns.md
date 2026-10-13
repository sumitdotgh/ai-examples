# FastAPI MCP Service Architecture Reference

This reference documents the Model Context Protocol (MCP) server and client implementations in `fastapi-mcp-api` and `test-mcp-client`.

## MCP Server Pattern

```python
# main.py
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP
from routes import weather_router

app = FastAPI(title="Weather MCP Service")

# Include modular routes
app.include_router(weather_router)

# Mount MCP endpoints
mcp = FastApiMCP(
    app,
    include_operations=[
        "get_all_weathers",
        "get_weather_info_by_city",
        "add_weather_info",
    ],
)
mcp.mount()
```

## Route & Schema Definition

```python
# routes/weather.py
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/weather", tags=["weather"])

class Weather(BaseModel):
    city: str
    temp: str

@router.get("/{city}", operation_id="get_weather_info_by_city", response_model=Weather)
async def get_weather(city: str):
    """Retrieve current weather by city."""
    return Weather(city=city, temp="22C")
```

## MCP Client Pattern (`httpx`)

```python
# mcp_client.py
import httpx
from pydantic import BaseModel

class Weather(BaseModel):
    city: str
    temp: str

class MCPClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url)

    async def get_weather_by_city(self, city: str) -> Weather:
        resp = await self.client.get(f"/weather/{city}")
        resp.raise_for_status()
        return Weather(**resp.json())

    async def close(self):
        await self.client.aclose()
```
