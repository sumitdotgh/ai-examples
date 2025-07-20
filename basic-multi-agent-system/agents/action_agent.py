from state import WeatherState
def action_agent(state: WeatherState) -> WeatherState:
    print("[Action] Performing actions on...")
    plan = state.plan
    action = f"Executed plan: {plan} -> Sunny 25°C in BLR"
    state.action = action
    print("[Executor] Executed plan, action:", action)
    return state
