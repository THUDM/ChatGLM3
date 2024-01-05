import abc

from typing import Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    calculation: str = Field(description="calculation to perform")


class Calculator(BaseTool, abc.ABC):
    name = "Calculator"
    description = "Useful for when you need to calculate math problems"
    args_schema: Type[BaseModel] = CalculatorInput

    def __init__(self):
        super().__init__()

    def _run(self, calculation: str) -> str:
        calculation = calculation.replace("^", "**")
        if "sqrt" in calculation:
            calculation = calculation.replace("sqrt", "math.sqrt")
        elif "log" in calculation:
            calculation = calculation.replace("log", "math.log")
        return eval(calculation)
