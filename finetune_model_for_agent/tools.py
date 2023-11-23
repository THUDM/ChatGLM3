import requests
import io
import base64
import os
from PIL import Image
from typing import Optional

from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain import LLMChain, PromptTemplate
from langchain.base_language import BaseLanguageModel
import re, random
from hashlib import md5

class APITool(BaseTool):
    name: str = ""
    description: str = ""
    url: str = ""

    def _call_api(self, query):
        raise NotImplementedError("subclass needs to overwrite this method")

    def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return self._call_api(query)

    async def _arun(
            self,
            query: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("APITool does not support async")

class functional_Tool(BaseTool):
    name: str = ""
    description: str = ""
    url: str = ""

    def _call_func(self, query):
        raise NotImplementedError("subclass needs to overwrite this method")

    def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return self._call_func(query)

    async def _arun(
            self,
            query: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("APITool does not support async")

# search tool #
class SearchTool(APITool):
    llm: BaseLanguageModel

    name = "搜索问答"
    description = "根据用户问题搜索最新的结果，并返回Json格式的结果"

    # search params
    google_api_key: str
    google_cse_id: str
    url = "https://www.googleapis.com/customsearch/v1"
    top_k = 2

    # QA params
    qa_template = """
    请根据下面带```分隔符的文本来回答问题。
    通过Search，如果该文本中没有相关内容可以回答问题，请直接回复：“抱歉，通过Search该问题需要更多上下文信息。”
    ```{text}```
    问题：{query}
    """
    prompt = PromptTemplate.from_template(qa_template)
    llm_chain: LLMChain = None

    def _call_api(self, query):
        self.get_llm_chain()
        context = self.get_search_result(query)
        resp = self.llm_chain.predict(text=context, query=query)
        return resp

    def get_search_result(self, query):
        data = {"key": self.google_api_key,
                "cx": self.google_cse_id,
                "q": query,
                "lr": "lang_zh-CN"}
        results = requests.get(self.url, params=data).json()
        results = results.get("items", [])[:self.top_k]
        snippets = []
        if len(results) == 0:
            return("No Search Result was found")
        for result in results:
            print("result:", result)
            text = ""
            if "title" in result:
                text += result["title"] + "。"
            if "snippet" in result:
                text += result["snippet"]
            snippets.append(text)
        return("\n\n".join(snippets))

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

class Text_classification_Tool(functional_Tool):
    llm: BaseLanguageModel

    name = "文本分类"
    description = "用户输入句子，完成文本分类"

    # QA params
    qa_template = """
    请根据下面带```分隔符的文本来回答问题。
    ```{text}```
    问题：{query}
    """
    prompt = PromptTemplate.from_template(qa_template)
    llm_chain: LLMChain = None

    def _call_func(self, query) -> str:
        self.get_llm_chain()
        context = "Instruction: 你是一个非常厉害的[词条名称]多层级分类模型"
        resp = self.llm_chain.predict(text=context, query=query)
        return resp

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
