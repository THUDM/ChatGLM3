# 工具调用
本文档将介绍如何使用 ChatGLM3-6B 进行工具调用。目前只有 ChatGLM3-6B 模型支持工具调用，而 ChatGLM3-6B-Base 和 ChatGLM3-6B-32K 模型不支持。

## 构建 System Prompt
这里以两个工具调用为例，首先准备好要构建的数据的描述信息。

```python
tools = [
    {
        "name": "track",
        "description": "追踪指定股票的实时价格",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "description": "需要追踪的股票代码"
                }
            },
            "required": ['symbol']
        }
    },
    {
        "name": "text-to-speech",
        "description": "将文本转换为语音",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "description": "需要转换成语音的文本"
                },
                "voice": {
                    "description": "要使用的语音类型（男声、女声等）"
                },
                "speed": {
                    "description": "语音的速度（快、中等、慢等）"
                }
            },
            "required": ['text']
        }
    }
]
system_info = {"role": "system", "content": "Answer the following questions as best as you can. You have access to the following tools:", "tools": tools}
```
请确保工具的定义格式与例子中一致以获得最优的性能

## 提出问题
注意：目前 ChatGLM3-6B 的工具调用只支持通过 `chat` 方法，不支持  `stream_chat` 方法。
```python
history = [system_info]
query = "帮我查询股票10111的价格"
response, history = model.chat(tokenizer, query, history=history)
print(response)
```
这里期望得到的输出为
```json
{"name": "track", "parameters": {"symbol": "10111"}}
```
这表示模型需要调用工具 `track`，并且需要传入参数 `symbol`。

## 调用工具，生成回复
这里需要自行实现调用工具的逻辑。假设已经得到了返回结果，将结果以 json 格式返回给模型并得到回复。
```python
result = json.dumps({"price": 12412}, ensure_ascii=False)
response, history = model.chat(tokenizer, result, history=history, role="observation")
print(response)
```
这里 `role="observation"` 表示输入的是工具调用的返回值而不是用户输入，不能省略。

期望得到的输出为
```
根据您的查询，经过API的调用，股票10111的价格是12412。
```

这表示本次工具调用已经结束，模型根据返回结果生成回复。对于比较复杂的问题，模型可能需要进行多次工具调用。这时，可以根据返回的 `response` 是 `str` 还是 `dict` 来判断返回的是生成的回复还是工具调用请求。