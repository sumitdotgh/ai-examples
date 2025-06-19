from typing import List
from .weather_repository import WeatherRepository
from models import Weather


class InMemoryWeatherRepository(WeatherRepository):
    __weathers = {}

    def get_by_city(self, city: str) -> Weather:
        """Method to return city specific weather"""        
        if city not in self.__weathers:
            raise Exception("No city found")

        return self.__weathers[city]

    def add(self, weather: Weather) -> Weather:
        """Method to add city specific weather"""        
        self.__weathers[weather.city] = weather
        return weather

    def get_all(self) -> List[Weather]:
        """Method to return weathers of all city"""        
        return list(self.__weathers.values())
