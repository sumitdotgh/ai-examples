from fastapi import FastAPI
from fastapi_mcp import FastApiMCP # pylint: disable=import-error
from routes import weather

app = FastAPI()

app.include_router(weather.router)

mcp = FastApiMCP(
    app,
    include_operations=[
        "get_all_weathers",
        "get_weather_info_by_city",
        "add_weather_info",
    ],
)

# Mounting the MCP server
mcp.mount()
