from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

from state import WeatherState
from agents.controller_agent import controller_agent
from agents.planner_agent import planner_agent
from agents.action_agent import action_agent
    

# Wrap your function using RunnableLambda
controller_node = RunnableLambda(controller_agent)
planner_node = RunnableLambda(planner_agent)
action_node = RunnableLambda(action_agent)

def build_weather_graph():
    builder = StateGraph(WeatherState)

    builder.add_node("controller", controller_node)
    builder.add_node("planner", planner_node)
    builder.add_node("action", action_node)

    builder.set_entry_point("controller")
    builder.add_edge("controller", "planner")
    builder.add_edge("planner", "action")
    builder.add_edge("action", END)

    return builder.compile()
