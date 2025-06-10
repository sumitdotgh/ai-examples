from fastapi import Query
from fastapi import APIRouter
from models import WeatherResponse


router = APIRouter()

@router.get("/weather", response_model=WeatherResponse, operation_id="get_weather_info")
async def get_weather(city: str = Query(..., example="Bangalore")):
    
    weather_info = WeatherResponse(
        city=city,
        temperature="10 degree celsius",
        description=f"{city} weather"
    )

    return weather_info