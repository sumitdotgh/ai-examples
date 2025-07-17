from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from state import WeatherState
from pydantic import SecretStr
import os

load_dotenv()

ENDPOINT = "https://models.github.ai/inference"
MODEL = "openai/gpt-4.1"
TOKEN = os.environ["GPT_4_1_MODEL_GITHUB_TOKEN"]
llm = ChatOpenAI(base_url=ENDPOINT, api_key=SecretStr(TOKEN), model=MODEL)

# Optional: prompt template for planning
PLANNER_PROMPT = ChatPromptTemplate.from_template("""
You are a smart planner agent for a weather system.
Based on the user query, generate a high-level plan.

User Query: {query}

Plan:
""")


def planner_agent(state: WeatherState) -> WeatherState:
    # Prepare the prompt
    prompt = PLANNER_PROMPT.format_messages(query=state.input_query)
    
    # Call the LLM
    response = llm.invoke(prompt)
    
    state.plan = response.content.strip() # type: ignore
    
    print("[Planner] Generated plan:", state.plan)

    return state
