import uvicorn
import asyncio

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentCapabilities
)
from agent_executor import (
    WeatherAgentExecutor
)

def main():
    
    skill = AgentSkill(
        id='weather_agent',
        name='Weather Agent',
        description='Share weather specific information',
        tags=['weather info'],
        examples=['What is the weather of blr?'],
    )
    
    agent_card = AgentCard(
        name="Weather agent",
        description="A simple agent that returns weather information",
        url="http://localhost:9999",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[skill],
        version="1.0.0",
        capabilities=AgentCapabilities()
    )

    request_handler = DefaultRequestHandler(
        agent_executor=WeatherAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,        
    )

    uvicorn.run(server.build(), host='0.0.0.0', port=9999)


if __name__ == '__main__':

    main()