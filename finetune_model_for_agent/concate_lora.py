from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
import torch
import os

model_name_or_path = "/opt/tiger/xuxuanwen/personal/LLaMA-Efficient-Tuning/meta-llama/llama-2-7b-extent/"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='auto', torch_dtype=torch.bfloat16)#.half().cuda()

peft_model_id = "/opt/tiger/xuxuanwen/personal/LLaMA-Efficient-Tuning/path_to_pt_checkpoint_2021-49_zh_head_000x_lora5000/checkpoint-39000"
model = PeftModel.from_pretrained(model, peft_model_id)
model = model.eval()

# 合并lora
model_merge = model.merge_and_unload()
merge_lora_model_path = "test_merge_dir"

model_merge.save_pretrained(merge_lora_model_path, max_shard_size="2GB")
tokenizer.save_pretrained(merge_lora_model_path)
