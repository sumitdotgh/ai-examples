from typing import List
from fastapi import Query
from fastapi import APIRouter
from models import Weather
from repositories.in_memory_weather_repository import InMemoryWeatherRepository

router = APIRouter()
weather_repository = InMemoryWeatherRepository()

@router.get("/weathers", response_model=List[Weather], operation_id="get_all_weathers")
async def get_all_weathers():    
    return weather_repository.get_all()

@router.get("/weathers/{city}", response_model=Weather, operation_id="get_weather_info_by_city")
async def get_weather_by_city(city: str):    
    return weather_repository.get_by_city(city=city)

@router.post("/weathers", response_model=Weather, operation_id="add_weather_info")
async def add_weather(payload: Weather):    
    return weather_repository.add(weather=payload)