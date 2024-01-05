import abc
from typing import Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class DistanceConversionInput(BaseModel):
    distance: float = Field(description="The numerical value of the distance to convert")
    unit: str = Field(description="The current unit of the distance (m, km, or feet)")
    to_unit: str = Field(description="The target unit to convert the distance into (m, km, or feet)")


class DistanceConverter(BaseTool, abc.ABC):
    name = "DistanceConverter"
    description = "Converts distance between meters, kilometers, and feet"
    args_schema: Type[BaseModel] = DistanceConversionInput

    def __init__(self):
        super().__init__()

    def _run(self, distance: float, unit: str, to_unit: str) -> str:
        unit_conversions = {
            "m_to_km": 0.001,
            "km_to_m": 1000,
            "feet_to_m": 0.3048,
            "m_to_feet": 3.28084,
            "km_to_feet": 3280.84,
            "feet_to_km": 0.0003048
        }

        if unit == to_unit:
            return f"{distance} {unit} is equal to {distance} {to_unit}"

        if unit == "km":
            distance *= unit_conversions["km_to_m"]
        elif unit == "feet":
            distance *= unit_conversions["feet_to_m"]

        if to_unit == "km":
            converted_distance = distance * unit_conversions["m_to_km"]
        elif to_unit == "feet":
            converted_distance = distance * unit_conversions["m_to_feet"]
        else:
            converted_distance = distance  # already in meters if this block is reached

        return f"{distance} {unit} is equal to {converted_distance} {to_unit}"
