from langchain.agents import AgentExecutor
from langchain.llms import AzureOpenAI
import openai
from llm import ChatGLM3
from tools import SearchTool, Text_classification_Tool
from agent import IntentAgent
import copy
import os

# google search api ley
GOOGLE_API_KEY = ""
GOOGLE_CSE_ID = ""

if __name__ == "__main__":
    model_path = "./model"
    llm_ori = ChatGLM3(model_path=model_path)
    llm_text_cls = ChatGLM3(model_path=model_path)

    llm_ori.load_model()
    llm_text_cls.load_model_from_checkpoint(checkpoint="text_classification")

    tools = [SearchTool(llm=llm_ori, google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID),
            Text_classification_Tool(llm=llm_text_cls)]

    agent = IntentAgent(tools=tools, llm=llm_ori)
    agent_exec = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, max_iterations=1)
    agent_exec.run("Input:《蝴蝶的翅膀》是一首由韦展辉作词、作曲并演唱的歌曲")
