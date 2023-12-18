import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from client import get_client
from conversation import postprocess_text, preprocess_text, Conversation, Role

client = get_client()


# Append a conversation into history, while show it in a new markdown block
def append_conversation(
        conversation: Conversation,
        history: list[Conversation],
        placeholder: DeltaGenerator | None = None,
) -> None:
    history.append(conversation)
    conversation.show(placeholder)


def main(
        prompt_text: str,
        system_prompt: str,
        top_p: float = 0.8,
        temperature: float = 0.95,
        repetition_penalty: float = 1.0,
        max_new_tokens: int = 1024,
        retry: bool = False
):
    placeholder = st.empty()
    with placeholder.container():
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    if prompt_text == "" and retry == False:
        print("\n== Clean ==\n")
        st.session_state.chat_history = []
        return

    history: list[Conversation] = st.session_state.chat_history
    for conversation in history:
        conversation.show()

    if retry:
        print("\n== Retry ==\n")
        last_user_conversation_idx = None
        for idx, conversation in enumerate(history):
            if conversation.role == Role.USER:
                last_user_conversation_idx = idx
        if last_user_conversation_idx is not None:
            prompt_text = history[last_user_conversation_idx].content
            del history[last_user_conversation_idx:]


    if prompt_text:
        prompt_text = prompt_text.strip()
        append_conversation(Conversation(Role.USER, prompt_text), history)
        placeholder = st.empty()
        message_placeholder = placeholder.chat_message(name="assistant", avatar="assistant")
        markdown_placeholder = message_placeholder.empty()

        output_text = ''
        for response in client.generate_stream(
                system_prompt,
                tools=None,
                history=history,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=[str(Role.USER)],
                repetition_penalty=repetition_penalty,
        ):
            token = response.token
            if response.token.special:
                print("\n==Output:==\n", output_text)
                match token.text.strip():
                    case '<|user|>':
                        break
                    case _:
                        st.error(f'Unexpected special token: {token.text.strip()}')
                        break
            output_text += response.token.text
            markdown_placeholder.markdown(postprocess_text(output_text + '▌'))

        append_conversation(Conversation(
            Role.ASSISTANT,
            postprocess_text(output_text),
        ), history, markdown_placeholder)