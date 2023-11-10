import os
import platform
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).cuda()
# 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
# from utils import load_model_on_gpus
# model = load_model_on_gpus("THUDM/chatglm3-6b", num_gpus=2)
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM3-6B：{response}"
    return prompt


tools = [{'name': 'track', 'description': '追踪指定股票的实时价格', 'parameters': {'type': 'object', 'properties': {'symbol': {'description': '需要追踪的股票代码'}}, 'required': []}}, {'name': '/text-to-speech', 'description': '将文本转换为语音', 'parameters': {'type': 'object', 'properties': {'text': {'description': '需要转换成语音的文本'}, 'voice': {'description': '要使用的语音类型（男声、女声等）'}, 'speed': {'description': '语音的速度（快、中等、慢等）'}}, 'required': []}}, {'name': '/image_resizer', 'description': '调整图片的大小和尺寸', 'parameters': {'type': 'object', 'properties': {'image_file': {'description': '需要调整大小的图片文件'}, 'width': {'description': '需要调整的宽度值'}, 'height': {'description': '需要调整的高度值'}}, 'required': []}}, {'name': '/foodimg', 'description': '通过给定的食品名称生成该食品的图片', 'parameters': {'type': 'object', 'properties': {'food_name': {'description': '需要生成图片的食品名称'}}, 'required': []}}]
system_item = {"role": "system",
               "content": "Answer the following questions as best as you can. You have access to the following tools:",
               "tools": tools}

def main():
    past_key_values, history = None, [system_item]
    role = "user"
    global stop_stream
    print("欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：") if role == "user" else input("\n结果：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None,  [system_item]
            role = "user"
            os.system(clear_command)
            print("欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        print("\nChatGLM：", end="")
        response, history = model.chat(tokenizer, query, history=history, role=role)
        print(response, end="", flush=True)
        print("")
        if isinstance(response, dict):
            role = "observation"


if __name__ == "__main__":
    main()