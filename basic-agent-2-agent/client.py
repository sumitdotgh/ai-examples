import asyncio
import uuid

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    Message,
    Role,
    Part,
    TextPart,
    MessageSendParams,
    SendMessageRequest,
)

PUBLIC_AGENT_CARD_PATH = "/.well-known/agent.json"
BASE_URL = "http://localhost:9999"


async def main() -> None:

    try:

        # intilized an http async context
        async with httpx.AsyncClient() as httpx_client:

            # Initialize A2ACardResolver
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=BASE_URL,
            )

            # Fetched public agent card
            public_agent_card = await resolver.get_agent_card()

            # Initialized a2a client
            a2a_client = A2AClient(
                httpx_client=httpx_client, agent_card=public_agent_card
            )

            # Preparing message to send
            message_payload = Message(
                messageId=str(uuid.uuid4()),
                role=Role.user,
                parts=[Part(root=TextPart(text="What's the weather in BLR?"))],
            )

            # Sending message
            request = SendMessageRequest(
                id=str(uuid.uuid4()), params=MessageSendParams(message=message_payload)
            )

            # Got back the response
            response = await a2a_client.send_message(request)
            print(response.model_dump(mode="json", exclude_none=True))

    except Exception as e:
        raise e


if __name__ == "__main__":
    asyncio.run(main())
