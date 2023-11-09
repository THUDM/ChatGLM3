import abc
import math
from typing import Any

from langchain.tools import BaseTool


class Calculator(BaseTool, abc.ABC):
    name = "Calculator"
    description = "Useful for when you need to answer questions about math"

    def __init__(self):
        super().__init__()

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        # 用例中没有用到 arun 不予具体实现
        pass


    def _run(self, para: str) -> str:
        para = para.replace("^", "**")
        if "sqrt" in para:
            para = para.replace("sqrt", "math.sqrt")
        elif "log" in para:
            para = para.replace("log", "math.log")
        return eval(para)


if __name__ == "__main__":
    calculator_tool = Calculator()
    result = calculator_tool.run("sqrt(2) + 3")
    print(result)
