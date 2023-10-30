# Tool Invocation
This document will introduce how to use the ChatGLM3-6B for tool invocation. Currently, only the ChatGLM3-6B model supports tool invocation, while the ChatGLM3-6B-Base and ChatGLM3-6B-32K models do not support it.

## Building System Prompt
Here are two examples of tool invocation. First, prepare the description information of the data to be built.

```python
tools = [
    {
        "name": "track",
        "description": "Track the real-time price of a specified stock",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "description": "The stock code that needs to be tracked"
                }
            },
            "required": ['symbol']
        }
    },
    {
        "name": "text-to-speech",
        "description": "Convert text to speech",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "description": "The text that needs to be converted into speech"
                },
                "voice": {
                    "description": "The type of voice to use (male, female, etc.)"
                },
                "speed": {
                    "description": "The speed of the speech (fast, medium, slow, etc.)"
                }
            },
            "required": ['text']
        }
    }
]
system_info = {"role": "system", "content": "Answer the following questions as best as you can. You have access to the following tools:", "tools": tools}
```

Please ensure that the definition format of the tool is consistent with the example to obtain optimal performance.

## Asking Questions
Note: Currently, the tool invocation of ChatGLM3-6B only supports the `chat` method and does not support the `stream_chat` method.
```python
history = [system_info]
query = "Help me inquire the price of stock 10111"
response, history = model.chat(tokenizer, query, history=history)
print(response)
```
The expected output here is
```json
{"name": "track", "parameters": {"symbol": "10111"}}
```
This indicates that the model needs to call the tool `track`, and the parameter `symbol` needs to be passed in.

## Invoke Tool, Generate Response
Here, you need to implement the logic of calling the tool yourself. Assuming that the return result has been obtained, return the result to the model in json format and get a response.
```python
result = json.dumps({"price": 12412}, ensure_ascii=False)
response, history = model.chat(tokenizer, result, history=history, role="observation")
print(response)
```
Here `role="observation"` indicates that the input is the return value of the tool invocation rather than user input, and it cannot be omitted.

The expected output is
```
Based on your query, after the API call, the price of stock 10111 is 12412.
```

This indicates that this tool invocation has ended, and the model generates a response based on the return result. For more complex questions, the model may need to make multiple tool invocations. At this time, you can judge whether the returned `response` is `str` or `dict` to determine whether the return is a generated response or a tool invocation request.