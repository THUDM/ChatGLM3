"""
This script demonstrates how to use the `bad_words_ids` argument in the context of a conversational AI model to filter out unwanted words or phrases from the model's responses. It's designed to showcase a fundamental method of content moderation within AI-generated text, particularly useful in scenarios where maintaining the decorum of the conversation is essential.

Usage:

- Interact with the model by typing queries. The model will generate responses while avoiding the specified bad words.
- Use 'clear' to clear the conversation history and 'stop' to exit the program.

Requirements:
- The script requires the Transformers library and an appropriate model checkpoint.

Note: The `bad_words_ids` feature is an essential tool for controlling the output of language models, particularly in user-facing applications where content moderation is crucial.
"""
import os
import platform

from transformers import AutoTokenizer, AutoModel

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

welcome_prompt = "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"

# probability tensor contains either `inf`, `nan` or element < 0

bad_words = ["你好", "ChatGLM"]
bad_word_ids = [tokenizer.encode(bad_word, add_special_tokens=False) for bad_word in bad_words]


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
        if query.strip().lower() == "stop":
            break
        if query.strip().lower() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print(welcome_prompt)
            continue

        # Attempt to generate a response
        try:
            print("\nChatGLM：", end="")
            current_length = 0
            response_generated = False
            for response, history, past_key_values in model.stream_chat(
                tokenizer, query, history=history, top_p=1,
                temperature=0.01,
                past_key_values=past_key_values,
                return_past_key_values=True,
                bad_words_ids=bad_word_ids  # assuming this is implemented correctly
            ):
                response_generated = True
                # Check if the response contains any bad words
                if any(bad_word in response for bad_word in bad_words):
                    print("我的回答涉嫌了 bad word")
                    break  # Break the loop if a bad word is detected

                # Otherwise, print the generated response
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
            if not response_generated:
                print("没有生成任何回答。")
        except RuntimeError as e:
            print(f"生成文本时发生错误：{e}，这可能是涉及到设定的敏感词汇")

        print("")

if __name__ == "__main__":
    main()