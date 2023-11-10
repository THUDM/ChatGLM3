from typing import List
from ChatGLM3 import ChatGLM3

from langchain.agents import load_tools
from Tool.Weather import Weather
from Tool.Calculator import Calculator

from langchain.agents import initialize_agent
from langchain.agents import AgentType


def run_tool(tools, llm, prompt_chain: List[str]):
    loaded_tolls = []
    for tool in tools:
        if isinstance(tool, str):
            loaded_tolls.append(load_tools([tool], llm=llm)[0])
        else:
            loaded_tolls.append(tool)
    agent = initialize_agent(
        loaded_tolls, llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    for prompt in prompt_chain:
        agent.run(prompt)


if __name__ == "__main__":
    model_path = "THUDM/chatglm3-6b"
    llm = ChatGLM3()
    llm.load_model(model_name_or_path=model_path)

    # arxiv: 单个工具调用示例 1
    run_tool(["arxiv"], llm, [
        "帮我查询GLM-130B相关工作"
    ])

    # weather: 单个工具调用示例 2
    run_tool([Weather()], llm, [
        "今天北京天气怎么样？",
        "What's the weather like in Shanghai today",
    ])

    # calculator: 单个工具调用示例 3
    run_tool([Calculator()], llm, [
        "12345679乘以54等于多少？",
        "3.14的3.14次方等于多少？",
        "根号2加上根号三等于多少？",
    ]),

    # arxiv + weather + calculator: 多个工具结合调用
    # run_tool([Calculator(), "arxiv", Weather()], llm, [
    #     "帮我检索GLM-130B相关论文",
    #     "今天北京天气怎么样？",
    #     "根号3减去根号二再加上4等于多少？",
    # ])