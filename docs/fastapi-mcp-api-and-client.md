## Fast API server and client

### Run the server

- Navigate to the `fastapi-mcp-api` directory.
- And execute the below command to run the server.

    ```sh
    uvicorn main:app --reload
    ```

### Run the client

- Navigate to the `test-mcp-client` directory.
- Then run the below command to execute the client.

    ```sh
    python main.py
    ```

## Visual Explaination

```mermaid
graph TD
    subgraph Client Side
        MainScript[main.py]
        MCPClient[MCPClient httpx]
    end

    subgraph FastAPI App
        App[FastAPI App]
        Router[routes.weather]
        InMemoryRepo[InMemoryWeatherRepository]
        WeatherRepo[WeatherRepository]
        WeatherModel[Weather]
    end

    MainScript --> MCPClient
    MCPClient -->|POST /weathers| App
    MCPClient -->|GET /weathers| App
    MCPClient -->|GET /weathers/<'city'>| App

    App --> Router
    Router --> InMemoryRepo
    InMemoryRepo -->|implements| WeatherRepo
    Router --> WeatherModel
    InMemoryRepo --> WeatherModel

    WeatherModel -->|inherits| Pydantic[BaseModel]
```