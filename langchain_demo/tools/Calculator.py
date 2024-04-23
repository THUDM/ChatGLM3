import abc
import re
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

    def parameter_validation(self, para: str):
        """
        You can write your own parameter validation rules here,
        you can refer to the code given here.
        :param para:
        :return:
        """
        symbols = ["math", "sqrt", "log", "sin", "cos", "tan", "pi"]
        for sym in symbols:
            para = para.replace(sym, "")
        patten = re.compile("[+*/\-%\d()=\s.]{3,}")
        if re.findall(patten, para):
            return True

    def _run(self, calculation: str) -> str:
        calculation = calculation.replace("^", "**")
        if "sqrt" in calculation and "math" not in calculation:
            calculation = calculation.replace("sqrt", "math.sqrt")
        if "log" in calculation and "math" not in calculation:
            calculation = calculation.replace("log", "math.log")
        if "sin" in calculation and "math" not in calculation:
            calculation = calculation.replace("sin", "math.sin")
        if "cos" in calculation and "math" not in calculation:
            calculation = calculation.replace("cos", "math.cos")
        if "tan" in calculation and "math" not in calculation:
            calculation = calculation.replace("tan", "math.tan")
        if "pi" in calculation and "math" not in calculation:
            calculation = calculation.replace("pi", "math.pi")
        if "pI" in calculation and "math" not in calculation:
            calculation = calculation.replace("pI", "math.pi")
        if "PI" in calculation and "math" not in calculation:
            calculation = calculation.replace("PI", "math.pi")
        if "Pi" in calculation and "math" not in calculation:
            calculation = calculation.replace("Pi", "math.pi")
        return eval(calculation)
