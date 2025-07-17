from state import WeatherState
def controller_agent(state: WeatherState) -> WeatherState:
    print("[Controller] Received query:", state.input_query)    
    return state
