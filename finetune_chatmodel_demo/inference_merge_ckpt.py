import argparse
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import os

parser = argparse.ArgumentParser()

ckpt_dir = "/root/autodl-tmp/ChatGLM3/finetune_chatmodel_demggo/output/zts_pt-20231229-141644-128-2e-2"
model_dir = "/root/autodl-tmp/model_files/chatglm3-6b-32k"
merged_dir = "/root/autodl-tmp/ChatGLM3/finetune_chatmodelgg_demo/output/merged"

parser.add_argument("--pt-checkpoint", type=str,
                    default=ckpt_dir,
                    help="The checkpoint path")
parser.add_argument("--model", type=str,
                    default=model_dir,
                    help="main model weights")
parser.add_argument("--tokenizer", type=str, default=model_dir, help="main model weights")
parser.add_argument("--pt-pre-seq-len", type=int, default=128, help="The pre-seq-len used in p-tuning")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--max-new-tokens", type=int, default=128)

args = parser.parse_args()

# if args.tokenizer is None:
#     args.tokenizer = args.model

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, pre_seq_len=args.pt_pre_seq_len)
model = AutoModel.from_pretrained(args.model, config=config, trust_remote_code=True).cuda()
prefix_state_dict = torch.load(os.path.join(args.pt_checkpoint, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

### save model
model.save_pretrained(save_directory=merged_dir, state_dict=model.state_dict())
tokenizer.save_pretrained(save_directory=merged_dir)
