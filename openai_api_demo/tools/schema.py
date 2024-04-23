
"""
Description: You can customize the developed langchain tool overview information here,
just like the sample code already given in this script.
"""

from tools.Calculator import Calculator


tool_class = {
                'Calculator': Calculator,
                # 'track': Track
            }


# E.g: "```python\ntool_call(symbol='37*8+7/2')\n```"
tool_param_start_with = "```python\ntool_call"


tool_def = [
    {"name": "track", "description": "追踪指定股票的实时价格", "parameters": {"type": "object", "properties": {"symbol": {"description": "需要追踪的股票代码"}}, "required": []}},
    {"name": "Calculator", "description": "数学计算器，计算数学问题", "parameters": {"type": "object", "properties": {"symbol": {"description": "要计算的数学公式"}}, "required": []}}
    ]
