from abc import ABC, abstractmethod
from typing import List, Optional
from models import Weather


class WeatherRepository(ABC):
    """Weather repository class"""
    @abstractmethod
    def get_by_city(self, city: str) -> Optional[Weather]:
        """Abstract method to return get weather by city"""
        pass

    @abstractmethod
    def add(self, weather: Weather) -> None:
        """Abstract method to add weather by city"""
        pass

    @abstractmethod
    def get_all(self) -> List[Weather]:
        """Abstract method to get all weathers"""
        pass
