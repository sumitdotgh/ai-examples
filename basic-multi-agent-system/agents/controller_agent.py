import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from state import WeatherState
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

ENDPOINT = "https://models.github.ai/inference"
MODEL = "openai/gpt-4.1"
TOKEN = os.environ["GPT_4_1_MODEL_GITHUB_TOKEN"]
llm = ChatOpenAI(base_url=ENDPOINT, api_key=SecretStr(TOKEN), model=MODEL)

intent_prompt = PromptTemplate.from_template("""
You are a controller agent. Given a query, decide if it's about:
- 'weather'
- 'temperature'
- 'unknown'

Query: {query}
Intent:
""")


def controller_agent(state: WeatherState) -> WeatherState:
    print("[Controller] Received query:", state.input_query)
    prompt = intent_prompt.format(query=state.input_query)
    response = llm.invoke(prompt)
    intent = response.content.strip().lower() # type: ignore

    if "weather" in intent or "temperature" in intent:
        print(f"[Controller] Routing as weather query (intent: {intent})")
        return state
    else:
        raise ValueError(f"[Controller] Unknown intent detected: {intent}")
