import json

import openai
from loguru import logger

from tool_register import get_tools, dispatch_tool

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "xxx"


tools = get_tools()
system_info = {
    "role": "system",
    "content": "Answer the following questions as best as you can. You have access to the following tools:",
    "tools": list(tools.values()),
}


def main():
    messages = [
        system_info,
        {
            "role": "user",
            "content": "帮我查询北京的天气怎么样",
        }
    ]
    response = openai.ChatCompletion.create(
        model="chatglm3",
        messages=messages,
        temperature=0,
        return_function_call=True
    )
    function_call = json.loads(response.choices[0].message.content)
    logger.info(f"Function Call Response: {function_call}")

    tool_response = dispatch_tool(function_call["name"], function_call["parameters"])
    logger.info(f"Tool Call Response: {tool_response}")

    messages = response.choices[0].history  # 获取历史对话信息
    messages.append(
        {
            "role": "observation",
            "content": tool_response,  # 调用函数返回结果
        }
    )

    response = openai.ChatCompletion.create(
        model="chatglm3",
        messages=messages,
        temperature=0,
    )
    logger.info(response.choices[0].message.content)


if __name__ == "__main__":
    main()
