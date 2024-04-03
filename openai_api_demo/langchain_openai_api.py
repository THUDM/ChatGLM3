"""
This script is designed for interacting with a local GLM3 AI model using the `ChatGLM3` class
from the `langchain_community` library. It facilitates continuous dialogue with the GLM3 model.

1. Start the Local Model Service: Before running this script, you need to execute the `api_server.py` script
to start the GLM3 model's service.
2. Run the Script: The script includes functionality for initializing the LLMChain object and obtaining AI responses,
allowing the user to input questions and receive AI answers.

"""
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.llms.chatglm3 import ChatGLM3


def initialize_llm_chain(messages: list):
    template = "{input}"
    prompt = PromptTemplate.from_template(template)

    endpoint_url = "http://127.0.0.1:8000/v1/chat/completions"
    llm = ChatGLM3(
        endpoint_url=endpoint_url,
        max_tokens=8096,
        prefix_messages=messages,
        top_p=0.9,
        streaming=True, # Set to True for streaming completions

    )
    return LLMChain(prompt=prompt, llm=llm)


def get_ai_response(llm_chain, user_message):
    ai_response = llm_chain.invoke({"input": user_message})
    return ai_response


def continuous_conversation():
    messages = [
        SystemMessage(content="You are an intelligent AI assistant, named ChatGLM3."),
    ]
    while True:
        user_input = input("Human (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        llm_chain = initialize_llm_chain(messages=messages)
        ai_response = get_ai_response(llm_chain, user_input)
        print("ChatGLM3: ", ai_response["text"])
        messages += [
            HumanMessage(content=user_input),
            AIMessage(content=ai_response["text"]),
        ]


if __name__ == "__main__":
    continuous_conversation()
