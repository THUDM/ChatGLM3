import abc
import math
import re
from typing import Any
from langchain.tools import BaseTool


class Calculator(BaseTool, abc.ABC):
    name = "Calculator"
    description = "Useful for when you need to answer questions about math"

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

    def _run(self, para: str) -> str:
        para = para.replace("^", "**")
        if "sqrt" in para and "math" not in para:
            para = para.replace("sqrt", "math.sqrt")
        if "log" in para and "math" not in para:
            para = para.replace("log", "math.log")
        if "sin" in para and "math" not in para:
            para = para.replace("sin", "math.sin")
        if "cos" in para and "math" not in para:
            para = para.replace("cos", "math.cos")
        if "tan" in para and "math" not in para:
            para = para.replace("tan", "math.tan")
        if "pi" in para and "math" not in para:
            para = para.replace("pi", "math.pi")
        if "PI" in para and "math" not in para:
            para = para.replace("PI", "math.pi")
        if "Pi" in para and "math" not in para:
            para = para.replace("Pi", "math.pi")
        return eval(para)


if __name__ == "__main__":
    calculator_tool = Calculator()
    result = calculator_tool.run("37*8+7/2")
    print(result)
