import os
import requests

from typing import Type, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    location: str = Field(description="the location need to check the weather")


class Weather(BaseTool):
    name = "weather"
    description = "Use for searching weather at a specific location"
    args_schema: Type[BaseModel] = WeatherInput

    def __init__(self):
        super().__init__()

    def _run(self, location: str) -> dict[str, Any]:
        api_key = os.environ["SENIVERSE_KEY"]
        url = f"https://api.seniverse.com/v3/weather/now.json?key={api_key}&location={location}&language=zh-Hans&unit=c"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            weather = {
                "temperature": data["results"][0]["now"]["temperature"],
                "description": data["results"][0]["now"]["text"],
            }
            return weather
        else:
            raise Exception(
                f"Failed to retrieve weather: {response.status_code}")
