from model.modeling_chatglm import ChatGLMForConditionalGeneration
import torch
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import sys
import os
import time
import json
from cover_alpaca2jsonl import format_example

torch.set_default_tensor_type(torch.cuda.HalfTensor)
model = ChatGLMForConditionalGeneration.from_pretrained("model", trust_remote_code=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)

model_dir = sys.argv[1]
test_file = sys.argv[2]
max_infer_num = int(sys.argv[3])
peft_path = "{}/chatglm-lora.pt".format(model_dir)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=True,
    target_modules=['query_key_value'],
    r=8,
    lora_alpha=32, lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
if os.path.exists(peft_path):
    model.load_state_dict(torch.load(peft_path), strict=False)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

from cover_alpaca2jsonl import format_example

instructions = json.load(open(test_file))
with torch.no_grad():
    for idx, item in tqdm(enumerate(instructions[:max_infer_num])):
        time1 = time.time()
        feature = format_example(item)
        input_text = feature['context']
        label = feature['target']
        ids = tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids])
        input_ids = input_ids.to(0)
        out = model.generate(
            input_ids=input_ids,
            max_length=2048,
            repetition_penalty=1.2,
            do_sample=True,
            temperature=0.1
        )
        out_text = tokenizer.decode(out[0])
        print("out_text:", out_text.strip())
