"""
This script is an example of using the OpenAI API to create various interactions with a ChatGLM3 model.
It includes functions to:

1. Conduct a basic chat session, asking about weather conditions in multiple cities.
2. Initiate a simple chat in Chinese, asking the model to tell a short story.
3. Retrieve and print embeddings for a given text input.

Each function demonstrates a different aspect of the API's capabilities, showcasing how to make requests
and handle responses.
"""

from openai import OpenAI
import time

base_url = "http://127.0.0.1:8000/v1/"
client = OpenAI(api_key="EMPTY", base_url=base_url)


def function_chat():
    messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model="chatglm3-6b",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    if response:
        content = response.choices[0].message.content
        print(content)
    else:
        print("Error:", response.status_code)


def simple_chat(use_stream=True):
    messages = [
        {
            "role": "system",
            "content": "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's "
                       "instructions carefully. Respond using markdown.",
        },
        {
            "role": "user",
            "content": "你好，请你用生动的话语给我讲一个小故事吧"
        }
    ]
    response = client.chat.completions.create(
        model="chatglm3-6b",
        messages=messages,
        stream=use_stream,
        max_tokens=256,
        temperature=0.8,
        presence_penalty=1.1,
        top_p=0.8)
    if response:
        if use_stream:
            for chunk in response:
                print(chunk.choices[0].delta.content)
        else:
            content = response.choices[0].message.content
            print(content)
    else:
        print("Error:", response.status_code)


if __name__ == "__main__":
    simple_chat(use_stream=False)
    simple_chat(use_stream=True)
