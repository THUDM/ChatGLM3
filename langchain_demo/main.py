"""
This script demonstrates the use of the LangChain's StructuredChatAgent and AgentExecutor alongside various tools

The script utilizes the ChatGLM3 model, a large language model for understanding and generating human-like text.
The model is loaded from a specified path and integrated into the chat agent.

Tools:
- Calculator: Performs arithmetic calculations.
- Weather: Provides weather-related information based on input queries.
- DistanceConverter: Converts distances between meters, kilometers, and feet.

The agent operates in three modes:
1. Single Parameter without History: Uses Calculator to perform simple arithmetic.
2. Single Parameter with History: Uses Weather tool to answer queries about temperature, considering the
conversation history.
3. Multiple Parameters without History: Uses DistanceConverter to convert distances between specified units.
4. Single use Langchain Tool: Uses Arxiv tool to search for scientific articles.

Note:
The model calling tool fails, which may cause some errors or inability to execute. Try to reduce the temperature
parameters of the model, or reduce the number of tools, especially the third function.
The success rate of multi-parameter calling is low. The following errors may occur:

Required fields [type=missing, input_value={'distance': '30', 'unit': 'm', 'to': 'km'}, input_type=dict]

The model illusion in this case generates parameters that do not meet the requirements.
The top_p and temperature parameters of the model should be adjusted to better solve such problems.

Success example:

*****Action*****

{
    'action': 'weather',
    'action_input': {
        'location': '厦门'
        }
}

*****Answer*****

{
    'input': '厦门比北京热吗?',
    'chat_history': [HumanMessage(content='北京温度多少度'), AIMessage(content='北京现在12度')],
    'output': '根据最新的天气数据，厦门今天的气温为18度，天气晴朗。而北京今天的气温为12度。所以，厦门比北京热。'
}

****************

"""

import os

from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent, load_tools
from langchain_core.messages import AIMessage, HumanMessage

from ChatGLM3 import ChatGLM3
from tools.Calculator import Calculator
from tools.Weather import Weather
from tools.DistanceConversion import DistanceConverter

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')

if __name__ == "__main__":
    llm = ChatGLM3()
    llm.load_model(MODEL_PATH)
    prompt = hub.pull("hwchase17/structured-chat-agent")

    # for single parameter without history

    tools = [Calculator()]
    agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    ans = agent_executor.invoke({"input": "34 * 34"})
    print(ans)

    # for singe parameter with history

    tools = [Weather()]
    agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    ans = agent_executor.invoke(
        {
            "input": "厦门比北京热吗?",
            "chat_history": [
                HumanMessage(content="北京温度多少度"),
                AIMessage(content="北京现在12度"),
            ],
        }
    )
    print(ans)

    # for multiple parameters without history

    tools = [DistanceConverter()]
    agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    ans = agent_executor.invoke({"input": "how many meters in 30 km?"})

    print(ans)

    # for using langchain tools

    tools = load_tools(["arxiv"], llm=llm)
    agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    ans = agent_executor.invoke({"input": "Describe the paper about GLM 130B"})

    print(ans)
