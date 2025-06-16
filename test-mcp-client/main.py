import asyncio
from typing import List
from mcp_client.client import MCPClient, Weather

async def main():
    client = MCPClient(base_url="http://localhost:8000")

    # Added blr weather
    weather = Weather(city="blr",temp="10c")
    added_blr_weather: Weather = await client.add_weather(weather=weather)    
    print("Added blr weather:", added_blr_weather)

    # Added hyd weather    
    weather = Weather(city="hyd",temp="20c")
    added_hyd_weather: Weather = await client.add_weather(weather=weather)    
    print("Added hyd weather:", added_hyd_weather)

    # Fetch all weathers
    all_weather_response: List[Weather] = await client.get_all_weathers()
    print("Fetched all weathers", all_weather_response)
    
    # Fetch a specific weather
    fetched_blr_weather: Weather = await client.get_weather_by_city(city="blr")
    print("Fetched BLR weather:", fetched_blr_weather)

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
