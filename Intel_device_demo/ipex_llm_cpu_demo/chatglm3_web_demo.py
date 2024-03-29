"""
This script creates an interactive web demo for the ChatGLM3-6B model using Gradio,
a Python library for building quick and easy UI components for machine learning models.
It's designed to showcase the capabilities of the ChatGLM3-6B model in a user-friendly interface,
allowing users to interact with the model through a chat-like interface.

Usage:
- Run the script to start the Gradio web server.
- Interact with the model by typing questions and receiving responses.

Requirements:
- Gradio (required for 4.13.0 and later, 3.x is not support now) should be installed.

Note: The script includes a modification to the Chatbot's postprocess method to handle markdown to HTML conversion,
ensuring that the chat interface displays formatted text correctly.

"""

import os
import streamlit as st
from ipex_llm.transformers import AutoModel
from transformers import AutoTokenizer


st.set_page_config(
    page_title="ChatGLM3-6B+BigDL-LLM demo",
    page_icon=":robot:",
    layout="wide"
)

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')

@st.cache_resource
def get_model():
    model = AutoModel.from_pretrained(MODEL_PATH,
                                    load_in_4bit=True,
                                    trust_remote_code=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,
                                            trust_remote_code=True)
    return tokenizer, model

tokenizer, model = get_model()

if "history" not in st.session_state:
    st.session_state.history = []
if "past_key_values" not in st.session_state:
    st.session_state.past_key_values = None

max_length = st.sidebar.slider("max_length", 0, 32768, 8192, step=1)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.6, step=0.01)

buttonClean = st.sidebar.button("clearing session history", key="clean")
if buttonClean:
    st.session_state.history = []
    st.session_state.past_key_values = None
    st.rerun()

for i, message in enumerate(st.session_state.history):
    if message["role"] == "user":
        with st.chat_message(name="user", avatar="user"):
            st.markdown(message["content"])
    else:
        with st.chat_message(name="assistant", avatar="assistant"):
            st.markdown(message["content"])

with st.chat_message(name="user", avatar="user"):
    input_placeholder = st.empty()
with st.chat_message(name="assistant", avatar="assistant"):
    message_placeholder = st.empty()

prompt_text = st.chat_input("please enter your question.")

if prompt_text:

    input_placeholder.markdown(prompt_text)
    history = st.session_state.history
    past_key_values = st.session_state.past_key_values
    for response, history, past_key_values in model.stream_chat(
        tokenizer,
        prompt_text,
        history,
        past_key_values=past_key_values,
        max_length=max_length,
        top_p=top_p,
        temperature=temperature,
        return_past_key_values=True,
    ):
        message_placeholder.markdown(response)

    st.session_state.history = history
    st.session_state.past_key_values = past_key_values