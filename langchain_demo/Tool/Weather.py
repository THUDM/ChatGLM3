import os
from typing import Any

import requests
from langchain.tools import BaseTool


class Weather(BaseTool):
    name = "weather"
    description = "Use for searching weather at a specific location"

    def __init__(self):
        super().__init__()

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        # 用例中没有用到 arun 不予具体实现
        pass

    def get_weather(self, location):
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

    def _run(self, para: str) -> str:
        return self.get_weather(para)


if __name__ == "__main__":
    weather_tool = Weather()
    weather_info = weather_tool.run("成都")
    print(weather_info)
