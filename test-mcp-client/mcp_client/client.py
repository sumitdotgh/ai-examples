import httpx
from typing import List
from pydantic import BaseModel


class Weather(BaseModel):
    city: str
    temp: str


class MCPClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=self.base_url)

    async def get_all_weathers(self) -> List[Weather]:
        response = await self.client.get("/weathers")
        response.raise_for_status()
        return response.json()

    async def get_weather_by_city(self, city) -> Weather:
        response = await self.client.get(f"/weathers/{city}")
        response.raise_for_status()
        return Weather(**response.json())

    async def add_weather(self, weather: Weather) -> Weather:
        response = await self.client.post("/weathers", json=weather.model_dump())
        response.raise_for_status()
        return Weather(**response.json())

    async def close(self):
        await self.client.aclose()
