import argparse
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("--pt-checkpoint", type=str, default=None, help="The checkpoint path")
parser.add_argument("--model", type=str, default=None, help="main model weights")
parser.add_argument("--tokenizer", type=str, default=None, help="main model weights")
parser.add_argument("--pt-pre-seq-len", type=int, default=128, help="The pre-seq-len used in p-tuning")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--max-new-tokens", type=int, default=128)

args = parser.parse_args()

if args.tokenizer is None:
    args.tokenizer = args.model

if args.pt_checkpoint:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, pre_seq_len=128)
    model = AutoModel.from_pretrained(args.model, config=config, trust_remote_code=True)
    prefix_state_dict = torch.load(os.path.join(args.pt_checkpoint, "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)

model = model.to(args.device)

while True:
    prompt = input("Prompt:")
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(args.device)
    response = model.generate(input_ids=inputs["input_ids"], max_length=inputs["input_ids"].shape[-1] + args.max_new_tokens)
    response = response[0, inputs["input_ids"].shape[-1]:]
    print("Response:", tokenizer.decode(response, skip_special_tokens=True))