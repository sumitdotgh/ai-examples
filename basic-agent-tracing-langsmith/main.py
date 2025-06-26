"""Basic agent"""

import os
from dotenv import load_dotenv
from pydantic import SecretStr

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()

TOKEN = os.environ["GPT_4_1_MODEL_GITHUB_TOKEN"]
ENDPOINT = "https://models.github.ai/inference"
MODEL = "openai/gpt-4.1"
secret = SecretStr(TOKEN)

llm = ChatOpenAI(base_url=ENDPOINT, api_key=secret, model=MODEL)


# setup the tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    print("In add")
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    print("In subtract")
    return a - b


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a calculation assistant.
        Use your tools to correct calculation. If you do not have a tool to
        provide the right result, then mention that you don't have supported tools. 

        Return only the answer. e.g
        Human: What is 2 + 2?
        AI: 4
        """,
        ),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# setup the multiple tools
tools = [add, subtract]

# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

QUERY = "What is the result of subtracting 3 from 10?"
result = agent_executor.invoke({"input": QUERY})

print(result["output"])
