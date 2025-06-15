from abc import ABC, abstractmethod
from typing import List, Optional
from models import Weather

class WeatherRepository(ABC):
    @abstractmethod
    def get_by_city(self, city: str) -> Optional[Weather]:
        pass
    @abstractmethod
    def add(self, weather: Weather) -> None:
        pass
    @abstractmethod
    def get_all(self) -> List[Weather]:
        pass
