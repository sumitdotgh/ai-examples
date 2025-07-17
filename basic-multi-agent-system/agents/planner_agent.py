from state import WeatherState
def planner_agent(state: WeatherState) -> WeatherState:    
    query = state.input_query
    if "forecast" in query:
        plan = "Get weather forecast"
    elif "temperature" in query:
        plan = "Fetch current temperature"
    else:
        plan = "General weather lookup"
    state.plan = plan
    print("[Planner] Generated plan:", plan)
    return state

