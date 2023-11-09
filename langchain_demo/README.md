# README

## 模型配置

在 `main.py` 文件中，修改 `model_path = /path/to/chatglm3-6b` 路径，也可以填写 `THUDM/chatglm3-6b` 自动下载模型。

## 工具添加

### LangChain 已实现工具

参考 [langchain](https://python.langchain.com/docs/modules/agents/tools/) 工具相关函数，在 `main.py` 中导入工具模块，例如导入 `arxiv` 工具

```python
run_tool(["arxiv"], llm, [
    "帮我查询AgentTuning相关工作"
])
```

#### Calculator、Weather Tool配置 

如果你的 Python 环境中 `LangChain` 的版本低于  **`0.0.278`**  则需要在这两个预定义工具类中实现 `_arun` 方法
否则将会出现 
`TypeError: Can't instantiate abstract class Weather with abstract method _arun`

示例如下：
```python
class Weather(BaseTool):
    name = "weather"
    description = "Use for searching weather at a specific location"

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        # 用例中没有用到 arun 不予具体实现
        pass
```

运行 `main.py` 文件

```
python main.py
```

模型会因找不到 `arxiv` 工具的 yaml 文件描述而中断，需要用户手动构建 `./Tool/arxiv.yaml` 文件。工具可以用户自行描述，也可以参考 LangChain 对该工具的描述。

对 `arxiv` 这个例子而言，参考内容位于 `./Tool/arxiv_example.yaml` 文件，可参考该文件构建 `Tool/arxiv.yaml` 文件（最简单的方式修改名称即可），重新运行模型就能合理调用工具。

> 有些工具需要导入 API_KEY，按照 langchain 报错添加到环境变量即可。

### 自定义工具

如果用户想自定义工具，可以参考 `Tool/Weather.py` 以及 `Tool/Weather.yaml` 文件，重载新的 `Tool` 类，实现其对应的 `_run()` 方法，然后在 `main.py` 中导入该工具模块，例如导入 `Weather` 工具，即可以调用

```python
# 对同一个工具调用多次
# 需要 export SENIVERSE_KEY=<YOUR_API_KEY_HERE>
run_tool([Weather()], llm, [
        "今天北京天气怎么样？",
        "What's the weather like in Shanghai today",
])
```

## 多工具使用

可以将多个工具组装在一起让模型自动选择调用，例如

```python
run_tool([Calculator(), "arxiv", Weather()], llm, [
        "帮我检索GLM-130B相关论文",
        "今天北京天气怎么样？",
        "根号3减去根号二再加上4等于多少？",
])
```
