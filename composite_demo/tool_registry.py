from copy import deepcopy
import inspect
from pprint import pformat
import traceback
from types import GenericAlias
from typing import get_origin, Annotated

_TOOL_HOOKS = {}
_TOOL_DESCRIPTIONS = {}

def register_tool(func: callable):
    tool_name = func.__name__
    tool_description = inspect.getdoc(func).strip()
    python_params = inspect.signature(func).parameters
    tool_params = []
    for name, param in python_params.items():
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            raise TypeError(f"Parameter `{name}` missing type annotation")
        if get_origin(annotation) != Annotated:
            raise TypeError(f"Annotation type for `{name}` must be typing.Annotated")
        
        typ, (description, required) = annotation.__origin__, annotation.__metadata__
        typ: str = str(typ) if isinstance(typ, GenericAlias) else typ.__name__
        if not isinstance(description, str):
            raise TypeError(f"Description for `{name}` must be a string")
        if not isinstance(required, bool):
            raise TypeError(f"Required for `{name}` must be a bool")

        tool_params.append({
            "name": name,
            "description": description,
            "type": typ,
            "required": required
        })
    tool_def = {
        "name": tool_name,
        "description": tool_description,
        "params": tool_params
    }

    print("[registered tool] " + pformat(tool_def))
    _TOOL_HOOKS[tool_name] = func
    _TOOL_DESCRIPTIONS[tool_name] = tool_def

    return func

def dispatch_tool(tool_name: str, tool_params: dict) -> str:
    if tool_name not in _TOOL_HOOKS:
        return f"Tool `{tool_name}` not found. Please use a provided tool."
    tool_call = _TOOL_HOOKS[tool_name]
    try:
        ret = tool_call(**tool_params)  
    except:
        ret = traceback.format_exc()
    return str(ret)

def get_tools() -> dict:
    return deepcopy(_TOOL_DESCRIPTIONS)

# Tool Definitions

@register_tool
def random_number_generator(
    seed: Annotated[int, 'The random seed used by the generator', True], 
    range: Annotated[tuple[int, int], 'The range of the generated numbers', True],
) -> int:
    """
    Generates a random number x, s.t. range[0] <= x < range[1]
    """
    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer")
    if not isinstance(range, tuple):
        raise TypeError("Range must be a tuple")
    if not isinstance(range[0], int) or not isinstance(range[1], int):
        raise TypeError("Range must be a tuple of integers")

    import random
    return random.Random(seed).randint(*range)

@register_tool
def get_weather(
    city_name: Annotated[str, 'The name of the city to be queried', True],
) -> str:
    """
    Get the current weather for `city_name`
    """

    if not isinstance(city_name, str):
        raise TypeError("City name must be a string")

    key_selection = {
        "current_condition": ["temp_C", "FeelsLikeC", "humidity", "weatherDesc",  "observation_time"],
    }
    import requests
    try:
        resp = requests.get(f"https://wttr.in/{city_name}?format=j1")
        resp.raise_for_status()
        resp = resp.json()
        ret = {k: {_v: resp[k][0][_v] for _v in v} for k, v in key_selection.items()}
    except:
        import traceback
        ret = "Error encountered while fetching weather data!\n" + traceback.format_exc() 

    return str(ret)

if __name__ == "__main__":
    print(dispatch_tool("get_weather", {"city_name": "beijing"}))
    print(get_tools())
