import time
from ipex_llm.transformers import AutoModel
from transformers import AutoTokenizer

CHATGLM_V3_PROMPT_FORMAT = "\n{prompt}\n"

# Please specify the local path to the chatglm3-6b model

model_path = "D:\AI\ChatGLM3\model/chatglm3-6b/"

# Load the ChatGLM3-6B model and quantize it to INT4
model = AutoModel.from_pretrained(model_path,
                                  load_in_4bit=True,
                                  trust_remote_code=True)
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path,
                                          trust_remote_code=True)
# Prepare ChatGLM3 format prompt
prompt = CHATGLM_V3_PROMPT_FORMAT.format(prompt="Who are you?")
# Encode the prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")
st = time.time()
# Perform inference calculation and generate Tokens
output = model.generate(input_ids,max_new_tokens=32)
end = time.time()
# Decode the generated Tokens and display them
output_str = tokenizer.decode(output[0], skip_special_tokens=True)
print(f'Inference time: {end-st} s')
print('-'*20, 'Prompt', '-'*20)
print(prompt)
print('-'*20, 'Output', '-'*20)
print(output_str)
