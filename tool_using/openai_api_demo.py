import json

import openai
from colorama import init, Fore
from loguru import logger

from tool_register import get_tools, dispatch_tool

init(autoreset=True)
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "xxx"


system_info = {
    "role": "system",
    "content": "Answer the following questions as best as you can. You have access to the following tools:",
    "tools": get_tools(),
}


def tool_test(query, stream=False):
    messages = [
        system_info,
        {
            "role": "user",
            "content": query,
        }
    ]

    response = openai.ChatCompletion.create(
        model="chatglm3",
        messages=messages,
        temperature=0,
        return_function_call=True,
        stream=stream,
    )
    for _ in range(5):
        if not stream:
            if response.choices[0].finish_reason == "stop":
                reply = response["choices"][0]["message"]["content"]
                logger.info(f"Final Reply: {reply}")
                return

            elif response.choices[0].finish_reason == "function_call":
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

        else:
            output = ""
            for chunk in response:
                content = chunk.choices[0].delta.get("content", "")
                print(Fore.BLUE + content, end="", flush=True)
                output += content

                if chunk.choices[0].finish_reason == "stop":
                    return

                elif chunk.choices[0].finish_reason == "function_call":
                    print("\n")

                    function_call = chunk.choices[0].function_call.to_dict_recursive()
                    logger.info(f"Function Call Response: {function_call}")

                    tool_response = dispatch_tool(function_call["name"], function_call["parameters"])
                    logger.info(f"Tool Call Response: {tool_response}")

                    messages = chunk.choices[0].history  # 获取历史对话信息
                    messages.append(
                        {
                            "role": "observation",
                            "content": tool_response,  # 调用函数返回结果
                        }
                    )

                    break

        response = openai.ChatCompletion.create(
            model="chatglm3",
            messages=messages,
            temperature=0,
            return_function_call=True,
            stream=stream,
        )


if __name__ == "__main__":
    query = "帮我查询北京的天气怎么样"
    tool_test(query, stream=True)
