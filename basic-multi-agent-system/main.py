# main.py
from weather_graph import build_weather_graph
from state import WeatherState
if __name__ == "__main__":
    weather_graph = build_weather_graph()

    # Initial shared state
    input_state = WeatherState(input_query="What's the weather like in Bangalore?")    

    print("🔁 Running Multi-Agent Weather Graph...")
    final_state = weather_graph.invoke(input_state)

    print("✅ Final Action:")
    print(final_state.get("action"))
