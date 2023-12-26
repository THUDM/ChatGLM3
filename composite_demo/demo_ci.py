import base64
from io import BytesIO
import os
from pprint import pprint
import queue
import re
from subprocess import PIPE

import jupyter_client
from PIL import Image
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from client import get_client
from conversation import postprocess_text, preprocess_text, Conversation, Role

IPYKERNEL = os.environ.get('IPYKERNEL', 'chatglm3-demo')

SYSTEM_PROMPT = '你是一位智能AI助手，你叫ChatGLM，你连接着一台电脑，但请注意不能联网。在使用Python解决任务时，你可以运行代码并得到结果，如果运行结果有错误，你需要尽可能对代码进行改进。你可以处理用户上传到电脑上的文件，文件默认存储路径是/mnt/data/。'

client = get_client()


class CodeKernel(object):
    def __init__(self,
                 kernel_name='kernel',
                 kernel_id=None,
                 kernel_config_path="",
                 python_path=None,
                 ipython_path=None,
                 init_file_path="./startup.py",
                 verbose=1):

        self.kernel_name = kernel_name
        self.kernel_id = kernel_id
        self.kernel_config_path = kernel_config_path
        self.python_path = python_path
        self.ipython_path = ipython_path
        self.init_file_path = init_file_path
        self.verbose = verbose

        if python_path is None and ipython_path is None:
            env = None
        else:
            env = {"PATH": self.python_path + ":$PATH", "PYTHONPATH": self.python_path}

        # Initialize the backend kernel
        self.kernel_manager = jupyter_client.KernelManager(kernel_name=IPYKERNEL,
                                                           connection_file=self.kernel_config_path,
                                                           exec_files=[self.init_file_path],
                                                           env=env)
        if self.kernel_config_path:
            self.kernel_manager.load_connection_file()
            self.kernel_manager.start_kernel(stdout=PIPE, stderr=PIPE)
            print("Backend kernel started with the configuration: {}".format(
                self.kernel_config_path))
        else:
            self.kernel_manager.start_kernel(stdout=PIPE, stderr=PIPE)
            print("Backend kernel started with the configuration: {}".format(
                self.kernel_manager.connection_file))

        if verbose:
            pprint(self.kernel_manager.get_connection_info())

        # Initialize the code kernel
        self.kernel = self.kernel_manager.blocking_client()
        # self.kernel.load_connection_file()
        self.kernel.start_channels()
        print("Code kernel started.")

    def execute(self, code):
        self.kernel.execute(code)
        try:
            shell_msg = self.kernel.get_shell_msg(timeout=30)
            io_msg_content = self.kernel.get_iopub_msg(timeout=30)['content']
            while True:
                msg_out = io_msg_content
                ### Poll the message
                try:
                    io_msg_content = self.kernel.get_iopub_msg(timeout=30)['content']
                    if 'execution_state' in io_msg_content and io_msg_content['execution_state'] == 'idle':
                        break
                except queue.Empty:
                    break

            return shell_msg, msg_out
        except Exception as e:
            print(e)
            return None

    def execute_interactive(self, code, verbose=False):
        shell_msg = self.kernel.execute_interactive(code)
        if shell_msg is queue.Empty:
            if verbose:
                print("Timeout waiting for shell message.")
        self.check_msg(shell_msg, verbose=verbose)

        return shell_msg

    def inspect(self, code, verbose=False):
        msg_id = self.kernel.inspect(code)
        shell_msg = self.kernel.get_shell_msg(timeout=30)
        if shell_msg is queue.Empty:
            if verbose:
                print("Timeout waiting for shell message.")
        self.check_msg(shell_msg, verbose=verbose)

        return shell_msg

    def get_error_msg(self, msg, verbose=False) -> str | None:
        if msg['content']['status'] == 'error':
            try:
                error_msg = msg['content']['traceback']
            except:
                try:
                    error_msg = msg['content']['traceback'][-1].strip()
                except:
                    error_msg = "Traceback Error"
            if verbose:
                print("Error: ", error_msg)
            return error_msg
        return None

    def check_msg(self, msg, verbose=False):
        status = msg['content']['status']
        if status == 'ok':
            if verbose:
                print("Execution succeeded.")
        elif status == 'error':
            for line in msg['content']['traceback']:
                if verbose:
                    print(line)

    def shutdown(self):
        # Shutdown the backend kernel
        self.kernel_manager.shutdown_kernel()
        print("Backend kernel shutdown.")
        # Shutdown the code kernel
        self.kernel.shutdown()
        print("Code kernel shutdown.")

    def restart(self):
        # Restart the backend kernel
        self.kernel_manager.restart_kernel()
        # print("Backend kernel restarted.")

    def interrupt(self):
        # Interrupt the backend kernel
        self.kernel_manager.interrupt_kernel()
        # print("Backend kernel interrupted.")

    def is_alive(self):
        return self.kernel.is_alive()


def b64_2_img(data):
    buff = BytesIO(base64.b64decode(data))
    return Image.open(buff)


def clean_ansi_codes(input_string):
    ansi_escape = re.compile(r'(\x9B|\x1B\[|\u001b\[)[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', input_string)


def execute(code, kernel: CodeKernel) -> tuple[str, str | Image.Image]:
    res = ""
    res_type = None
    code = code.replace("<|observation|>", "")
    code = code.replace("<|assistant|>interpreter", "")
    code = code.replace("<|assistant|>", "")
    code = code.replace("<|user|>", "")
    code = code.replace("<|system|>", "")
    msg, output = kernel.execute(code)

    if msg['metadata']['status'] == "timeout":
        return res_type, 'Timed out'
    elif msg['metadata']['status'] == 'error':
        return res_type, clean_ansi_codes('\n'.join(kernel.get_error_msg(msg, verbose=True)))

    if 'text' in output:
        res_type = "text"
        res = output['text']
    elif 'data' in output:
        for key in output['data']:
            if 'text/plain' in key:
                res_type = "text"
                res = output['data'][key]
            elif 'image/png' in key:
                res_type = "image"
                res = output['data'][key]
                break

    if res_type == "image":
        return res_type, b64_2_img(res)
    elif res_type == "text" or res_type == "traceback":
        res = res

    return res_type, res


@st.cache_resource
def get_kernel():
    kernel = CodeKernel()
    return kernel


def extract_code(text: str) -> str:
    pattern = r'```([^\n]*)\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1][1]


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
        top_p: float = 0.2,
        temperature: float = 0.1,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 1024,
        truncate_length: int = 1024,
        retry: bool = False
):
    if 'ci_history' not in st.session_state:
        st.session_state.ci_history = []


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
        role = Role.USER
        append_conversation(Conversation(role, prompt_text), history)

        placeholder = st.container()
        message_placeholder = placeholder.chat_message(name="assistant", avatar="assistant")
        markdown_placeholder = message_placeholder.empty()

        for _ in range(5):
            output_text = ''
            for response in client.generate_stream(
                    system=SYSTEM_PROMPT,
                    tools=None,
                    history=history,
                    do_sample=True,
                    max_new_token=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop_sequences=[str(r) for r in (Role.USER, Role.OBSERVATION)],
                    repetition_penalty=repetition_penalty,
            ):
                token = response.token
                if response.token.special:
                    print("\n==Output:==\n", output_text)
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
                            message_placeholder = placeholder.chat_message(name="interpreter", avatar="assistant")
                            markdown_placeholder = message_placeholder.empty()
                            output_text = ''
                            continue
                        case '<|observation|>':
                            code = extract_code(output_text)

                            display_text = output_text.split('interpreter')[-1].strip()
                            append_conversation(Conversation(
                                Role.INTERPRETER,
                                postprocess_text(display_text),
                            ), history, markdown_placeholder)
                            message_placeholder = placeholder.chat_message(name="observation", avatar="user")
                            markdown_placeholder = message_placeholder.empty()
                            output_text = ''

                            with markdown_placeholder:
                                with st.spinner('Executing code...'):
                                    try:
                                        res_type, res = execute(code, get_kernel())
                                    except Exception as e:
                                        st.error(f'Error when executing code: {e}')
                                        return
                            print("Received:", res_type, res)
                            if truncate_length:
                                if res_type == 'text' and len(res) > truncate_length:
                                    res = res[:truncate_length] + ' [TRUNCATED]'

                            append_conversation(Conversation(
                                Role.OBSERVATION,
                                '[Image]' if res_type == 'image' else postprocess_text(res),
                                tool=None,
                                image=res if res_type == 'image' else None,
                            ), history, markdown_placeholder)
                            message_placeholder = placeholder.chat_message(name="assistant", avatar="assistant")
                            markdown_placeholder = message_placeholder.empty()
                            output_text = ''
                            break
                        case _:
                            st.error(f'Unexpected special token: {token.text.strip()}')
                            break
                output_text += response.token.text
                display_text = output_text.split('interpreter')[-1].strip()
                markdown_placeholder.markdown(postprocess_text(display_text + '▌'))
            else:
                append_conversation(Conversation(
                    Role.ASSISTANT,
                    postprocess_text(output_text),
                ), history, markdown_placeholder)
                return
    else:
        st.session_state.chat_history = []
