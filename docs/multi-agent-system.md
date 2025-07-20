## Basic Multi Agent System

## Execute the system

```sh
python main.py
```

### Visual Explaination

#### Flow Diagram

```mermaid
flowchart TD
    A[User submits weather query]
    B[main.py calls build_weather_graph]
    C[Controller Agent processes input]
    D{Is input weather-related?}
    E1[Planner Agent uses LLM to create plan]
    E2[Return error or fallback response]
    F[Plan stored in state]
    G[Action Agent executes the plan]
    H[Action stored in state]
    I[main.py prints final action]
    J[Final result returned to user]

    A --> B --> C --> D
    D -- Yes --> E1 --> F --> G --> H --> I --> J
    D -- No --> E2 --> J
```

#### Sequence Diagram 

```mermaid
sequenceDiagram
    participant User as User
    participant Main as main.py
    participant Controller as ControllerAgent
    participant Planner as PlannerAgent (LLM)
    participant Action as ActionAgent
    participant Result as Final State

    User->>Main: Start with query "What's the weather like in Bangalore?"
    Main->>Controller: Pass WeatherState(input_query)
    Controller->>Controller: Print and forward state
    Controller-->>Main: Return unchanged state

    Main->>Planner: Pass state.input_query
    Planner->>LLM: Generate plan from query
    LLM-->>Planner: Plan = "Check weather in Bangalore"
    Planner->>Planner: Update state.plan
    Planner-->>Main: Return updated state

    Main->>Action: Pass state.plan
    Action->>Action: Simulate execution (e.g., fetch weather)
    Action->>Action: Update state.action = "Executed plan → Sunny 25°C in BLR"
    Action-->>Main: Return updated state

    Main->>Result: Final State with action
    Result-->>User: Print final action response
```