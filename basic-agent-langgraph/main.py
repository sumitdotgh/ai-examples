import os
import logging

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic import BaseModel
from functools import wraps


from langchain_openai import ChatOpenAI

from langchain.tools import tool
from langchain.tools import StructuredTool

from langchain_core.prompts import ChatPromptTemplate

from langgraph.prebuilt import create_react_agent


load_dotenv()

logging.basicConfig(filename="basic-agent-langgraph.log", level=logging.INFO)

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-mini"
secret = SecretStr(token)

llm = ChatOpenAI(
    base_url=endpoint,
    api_key=secret,
    model=model
)

class Input(BaseModel):
    a: int
    b: int

# Logging all tool calls
def tool_logger(tool_fn):
    @wraps(tool_fn)
    def wrapper(*args, **kwargs):
        logging.info(f"[LOG] Calling tool {tool_fn} with args={args} kwargs={kwargs}")
        return tool_fn(*args, **kwargs)
    return wrapper

def make_logged_tool(fn,schema):    
    return StructuredTool.from_function(
        func=tool_logger(fn),
        name=fn.__name__,
        description=fn.__doc__,
        args_schema=schema,
    )


# setup the tools
@tool_logger
def add(a: int, b: int) -> int:
    """Add two numbers."""    
    logging.info("in add method")
    return a + b

@tool_logger
def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""    
    logging.info("in subtract method")
    return a - b

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a calculation assistant.
        Use your tools to correct calculation. If you do not have a tool to
        provide the right result, then mention that you don't have supported tools. 

        Return only the answer. e.g
        Human: What is addition of a=2 and b=2?
        AI: 4
        """),
        ("placeholder", "{messages}"),        
    ]
)

# setup the multiple tools with logging
tools = [make_logged_tool(add,Input), make_logged_tool(subtract,Input)]

#query = "Use the calculation assistant to subtract two values a=10, b=3"
query = "Use the calculation assistant for adding two numbers a=10, b=3"

# Latest langgrpah specific usage
langgraph_agent_executor = create_react_agent(model=llm, tools=tools, prompt=prompt, debug=False)

messages = langgraph_agent_executor.invoke({"messages": [("human", query)]})
print(
    {
        "input": query,
        "output": messages["messages"][-1].content,
    }
)