import os
import platform
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# for Mac Computer like M1
# You Need Use Pytorch compiled with Metal
# DEVICE = 'mps'

# for AMD gpu likes MI100 (Not Official Steady Support yet)
# You Need Use Pytorch compiled with ROCm
# DEVICE = 'cuda'

# for Intel gpu likes A770 (Not Official Steady Support yet)
# You Need Use Pytorch compiled with oneDNN and install intel-extension-for-pytorch
# import intel_extension_for_pytorch as ipex
# DEVICE = 'xpu'

# for Moore Threads gpu like MTT S80 (Not Official Steady Support yet)
# You Need Use Pytorch compiled with Musa
# DEVICE = 'musa'



tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
if 'cuda' in DEVICE: # AMD, NVIDIA GPU can use Half Precision
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE).eval()
else: # CPU, Intel GPU and other GPU can use Float16 Precision Only
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).float().to(DEVICE).eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

welcome_prompt = "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"


def build_prompt(history):
    prompt = welcome_prompt
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM3-6B：{response}"
    return prompt


def main():
    past_key_values, history = None, []
    global stop_stream
    print(welcome_prompt)
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print(welcome_prompt)
            continue
        print("\nChatGLM：", end="")
        current_length = 0
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history, top_p=1,
                                                                    temperature=0.01,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
        print("")


if __name__ == "__main__":
    main()
