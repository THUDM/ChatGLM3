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
2. Single Parameter with History: Uses Weather tool to answer queries about temperature, considering the conversation history.
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

*****Action****
{'action': 'Calculator', 'action_input': {'calculation': '34*34'}}
***************
{'input': '34 * 34', 'output': '34 * 34 = 1156'}
{'input': '厦门比北京热吗?', 'chat_history': ['北京的温度是多少度?', '北京的温度是4度。'], 'output': '经过查询，厦门的温度是21度，比北京更加温暖'}
*****Action****
{'action': 'DistanceConverter', 'action_input': {'distance': '30', 'unit': 'km', 'to_unit': 'm'}}
***************
{'input': 'how many meters in 30 km?', 'output': 'The answer is: 30000'}
*****Action****
{'action': 'arxiv', 'action_input': {'query': 'GLM 130B'}}
***************
{'input': 'Descirbe the paper about GLM 130B', 'output': 'The GLM-130B paper is about a bilingual pre-trained language model with 130 billion parameters, trained on both English and Chinese data. The authors propose a unique scaling property of GLM-130B that allows them to reach INT4 quantization without post-training, which makes it the first 100B-scale model to achieve this. The model shows significant outperformance on popular English benchmarks and consistently outperforms the largest Chinese language model. The authors also show that masked language models, which are trained to predict masked tokens in a sequence, often demonstrate inconsistencies, and they propose a self-ensemble algorithm to address this issue during the inference phase.'}

"""

import os

from langchain.agents import StructuredChatAgent, AgentExecutor, load_tools
from ChatGLM3 import ChatGLM3

from tools.Calculator import Calculator
from tools.Weather import Weather
from tools.DistanceConversion import DistanceConverter

HUMAN_MESSAGE_TEMPLATE = "{input}\n\nhistory:{chat_history}\n\n{agent_scratchpad}"

MODEL_PATH = os.environ.get('MODEL_PATH', '/share/home/zyx/Models/chatglm3-6b')

if __name__ == "__main__":
    llm = ChatGLM3()
    llm.load_model(MODEL_PATH)

    # for single parameter without history

    tools = [Calculator()]
    agent = StructuredChatAgent.from_llm_and_tools(llm=llm, tools=tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    ans = agent_executor.invoke({"input": "34 * 34"})
    print(ans)

    # for singe parameter with history

    tools = [Weather()]
    agent = StructuredChatAgent.from_llm_and_tools(llm=llm, tools=tools, human_message_template=HUMAN_MESSAGE_TEMPLATE)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    ans = agent_executor.invoke(
        {
            "input": "厦门比北京热吗?",
            "chat_history": [
                "北京的温度是多少度?",
                "北京的温度是4度。",
            ],
        }
    )
    print(ans)

    # for multiple parameters without history

    tools = [DistanceConverter()]
    agent = StructuredChatAgent.from_llm_and_tools(llm=llm, tools=tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    ans = agent_executor.invoke({"input": "how many meters in 30 km?"})

    print(ans)

    # for using langchain tools

    tools = load_tools(["arxiv"], llm=llm)
    agent = StructuredChatAgent.from_llm_and_tools(llm=llm, tools=tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    ans = agent_executor.invoke({"input": "Descirbe the paper about GLM 130B"})

    print(ans)
