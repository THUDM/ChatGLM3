
"""
Description: You can customize the developed langchain tool overview information here,
just like the sample code already given in this script.
"""


tool_param_start_with = "```python\ntool_call"


""" Fill this dictionary with the mapping from tool class names to tool classes that you defined.  
Like: 
         from tools.Calculator import Calculator
         tool_class = {"Calculator": Calculator, ...}

It is required that your customized tool class must define the format for the langchain tool 
and implement the parameter verification function in the class:
       parameter_validation(self, para: str) -> bool
       
Tool class definition reference: ChatGLM3/langchain_demo/tools.
"""
tool_class = {}


""" Describe your tool names and parameters in this dictionary.
Like:
  tool_def = [
    {"name": "Calculator", 
     "description": "数学计算器，计算数学问题", 
      "parameters": {"type": "object", "properties": {"symbol": {"description": "要计算的数学公式"}}, "required": []}
     },...
  ]
"""
tool_def = []
