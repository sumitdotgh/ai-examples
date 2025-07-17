from typing import Optional
from pydantic import BaseModel
class WeatherState(BaseModel):
    input_query: str                   # Original user query
    plan: Optional[str] = None         # Plan created by PlannerAgent    
    action: Optional[str] = None       # Result from ActionAgent        


def log_state(state: dict, stage: str):
    print(f"[{stage}] Current state snapshot: {state}")

