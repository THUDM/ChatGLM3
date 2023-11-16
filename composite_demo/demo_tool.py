import re
import yaml
from yaml import YAMLError

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from client import get_client
from conversation import postprocess_text, preprocess_text, Conversation, Role
from tool_registry import dispatch_tool, get_tools

MAX_LENGTH = 8192
TRUNCATE_LENGTH = 1024

EXAMPLE_TOOL = {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    }
}

client = get_client()

def tool_call(*args, **kwargs) -> dict:
    print("=== Tool call:")
    print(args)
    print(kwargs)
    st.session_state.calling_tool = True
    return kwargs

def yaml_to_dict(tools: str) -> list[dict] | None:
    try:
        return yaml.safe_load(tools)
    except YAMLError:
        return None

def extract_code(text: str) -> str:
    pattern = r'```([^\n]*)\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1][1]

# Append a conversation into history, while show it in a new markdown block
def append_conversation(
    conversation: Conversation,
    history: list[Conversation],
    placeholder: DeltaGenerator | None=None,
) -> None:
    history.append(conversation)
    conversation.show(placeholder)

def main(top_p: float, temperature: float, prompt_text: str, repetition_penalty: float):
    manual_mode = st.toggle('Manual mode',
        help='Define your tools in YAML format. You need to supply tool call results manually.'
    )

    if manual_mode:
        with st.expander('Tools'):
            tools = st.text_area(
                'Define your tools in YAML format here:',
                yaml.safe_dump([EXAMPLE_TOOL], sort_keys=False),
                height=400,
            )
        tools = yaml_to_dict(tools)

        if not tools:
            st.error('YAML format error in tools definition')
    else:
        tools = get_tools()

    if 'tool_history' not in st.session_state:
        st.session_state.tool_history = []
    if 'calling_tool' not in st.session_state:
        st.session_state.calling_tool = False

    history: list[Conversation] = st.session_state.tool_history

    for conversation in history:
        conversation.show()

    if prompt_text:
        prompt_text = prompt_text.strip()
        role = st.session_state.calling_tool and Role.OBSERVATION or Role.USER
        append_conversation(Conversation(role, prompt_text), history)
        st.session_state.calling_tool = False

        input_text = preprocess_text(
            None,
            tools,
            history,
        )
        print("=== Input:")
        print(input_text)
        print("=== History:")
        print(history)

        placeholder = st.container()
        message_placeholder = placeholder.chat_message(name="assistant", avatar="assistant")
        markdown_placeholder = message_placeholder.empty()

        for _ in range(5):
            output_text = ''
            for response in client.generate_stream(
                system=None,
                tools=tools,
                history=history,
                do_sample=True,
                max_length=MAX_LENGTH,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=[str(r) for r in (Role.USER, Role.OBSERVATION)],
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
            ):
                token = response.token
                if response.token.special:
                    print("=== Output:")
                    print(output_text)

                    match token.text.strip():
                        case '<|user|>':
                            append_conversation(Conversation(
                                Role.ASSISTANT,
                                postprocess_text(output_text),
                            ), history, markdown_placeholder)
                            return
                        # Initiate tool call
                        case '<|assistant|>':
                            append_conversation(Conversation(
                                Role.ASSISTANT,
                                postprocess_text(output_text),
                            ), history, markdown_placeholder)
                            output_text = ''
                            message_placeholder = placeholder.chat_message(name="tool", avatar="assistant")
                            markdown_placeholder = message_placeholder.empty()
                            continue
                        case '<|observation|>':
                            tool, *output_text = output_text.strip().split('\n')
                            output_text = '\n'.join(output_text)
                            
                            append_conversation(Conversation(
                                Role.TOOL,
                                postprocess_text(output_text),
                                tool,
                            ), history, markdown_placeholder)
                            message_placeholder = placeholder.chat_message(name="observation", avatar="user")
                            markdown_placeholder = message_placeholder.empty()
                            
                            try:
                                code = extract_code(output_text)
                                args = eval(code, {'tool_call': tool_call}, {})
                            except:
                                st.error('Failed to parse tool call')
                                return
                            
                            output_text = ''
                            
                            if manual_mode:
                                st.info('Please provide tool call results below:')
                                return
                            else:
                                with markdown_placeholder:
                                    with st.spinner(f'Calling tool {tool}...'):
                                        observation = dispatch_tool(tool, args)

                                if len(observation) > TRUNCATE_LENGTH:
                                    observation = observation[:TRUNCATE_LENGTH] + ' [TRUNCATED]'
                                append_conversation(Conversation(
                                    Role.OBSERVATION, observation
                                ), history, markdown_placeholder)
                                message_placeholder = placeholder.chat_message(name="assistant", avatar="assistant")
                                markdown_placeholder = message_placeholder.empty()
                                st.session_state.calling_tool = False
                                break
                        case _:
                            st.error(f'Unexpected special token: {token.text.strip()}')
                            return
                output_text += response.token.text
                markdown_placeholder.markdown(postprocess_text(output_text + '▌'))
            else:
                append_conversation(Conversation(
                    Role.ASSISTANT,
                    postprocess_text(output_text),
                ), history, markdown_placeholder)
                return
