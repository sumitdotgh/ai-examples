from fastapi import FastAPI
from fastapi_mcp import FastApiMCP
from routes import weather

app = FastAPI()

app.include_router(weather.router)

mcp = FastApiMCP(app, include_operations=["get_weather_info"])

# Mounting the MCP server
mcp.mount()