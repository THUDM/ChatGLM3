import argparse
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import os

merged_dir = "/root/autodl-tmp/ChatGLM3/finetune_chatmodel_demo/output/merged"

parser = argparse.ArgumentParser()
parser.add_argument("--pt-checkpoint", type=str, default=None, help="The checkpoint path")
# parser.add_argument("--pt-checkpoint", type=str, default="/root/autodl-tmp/ChatGLM3/finetune_chatmodel_demo/output/zts_pt-20231229-141644-128-2e-2", help="The checkpoint path")
# parser.add_argument("--model", type=str, default="/root/autodl-tmp/model_files/chatglm3-6b-32k", help="main model weights")
parser.add_argument("--model", type=str, default=merged_dir, help="main model weights")
# parser.add_argument("--model", type=str, default=None, help="main model weights")
parser.add_argument("--tokenizer", type=str, default=None, help="main model weights")
parser.add_argument("--pt-pre-seq-len", type=int, default=128, help="The pre-seq-len used in p-tuning")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--max-new-tokens", type=int, default=128)

args = parser.parse_args()

if args.tokenizer is None:
    args.tokenizer = args.model
print(">>> ", args.tokenizer)

if args.pt_checkpoint:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, pre_seq_len=args.pt_pre_seq_len)
    model = AutoModel.from_pretrained(args.model, config=config, trust_remote_code=True).cuda()
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

import json
from tqdm import tqdm
import re
import string


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


from collections import Counter


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


n = 1
f1_sum = 0
with open("/root/autodl-tmp/ChatGLM3/data/zts/zts_test.jsonl", 'r') as file:
    for line in file:
        # Parse each line as a JSON object
        json_object = json.loads(line.strip())

        # Process the JSON object as needed
        prompt = "你是中泰证券的客服系统。根据以下问题提供回答，如果无法回答则输出空。"  # prompt template
        prompt += json_object['prompt']
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(args.device)
        response = model.generate(input_ids=inputs["input_ids"],
                                  max_length=inputs["input_ids"].shape[-1] + args.max_new_tokens)
        response = response[0, inputs["input_ids"].shape[-1]:]

        print("Response gen:", tokenizer.decode(response, skip_special_tokens=True))
        print("ground truth:", json_object['response'])

        f1 = qa_f1_score(tokenizer.decode(response, skip_special_tokens=True), json_object['response'])
        print(">>>> f1=", f1, ",  avg F1=", f1_sum / n)
        n += 1
        f1_sum += f1
        print()
