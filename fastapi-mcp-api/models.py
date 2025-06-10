from pydantic import BaseModel, Field

class WeatherResponse(BaseModel):
    city: str
    temperature: str
    description: str