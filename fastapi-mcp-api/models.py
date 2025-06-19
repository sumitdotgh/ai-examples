from pydantic import BaseModel, Field


class Weather(BaseModel):
    city: str
    temp: str
