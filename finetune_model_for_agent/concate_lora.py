from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
import torch
import os

model_name_or_path = "./model"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='auto', torch_dtype=torch.bfloat16)#.half().cuda()

peft_model_id = "./output/checkpoint-39000"
model = PeftModel.from_pretrained(model, peft_model_id)
model = model.eval()

# 合并lora
model_merge = model.merge_and_unload()
merge_lora_model_path = "test_merge_dir"

model_merge.save_pretrained(merge_lora_model_path, max_shard_size="2GB")
tokenizer.save_pretrained(merge_lora_model_path)
