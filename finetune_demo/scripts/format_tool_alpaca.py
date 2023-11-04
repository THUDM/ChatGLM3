#! /usr/bin/env python

import json
from collections import Counter
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)

args = parser.parse_args()

with open(args.path) as f:
    data = json.load(f)

train_examples = []
err_count = 0
for setting in data:
    api_desc = [setting["NLDocumentation"]]
    for instance in setting["Instances"]:
        try:
            conv = [{
                "role": "user",
                "content": instance['input'],
            }]
            for step in instance['intermediate_steps']:
                tool_name, params, react = step[0]
                step_thought = react.split("Action:")[0].strip()
                observation = step[1]
                conv.append({
                    "role": "assistant",
                    "content": step_thought,
                })
                conv.append({
                    "role": "tool",
                    "name": tool_name,
                    "parameters": json.loads(params),
                    "observation": observation,
                })
            conv.append({
                "role": "assistant",
                "content": instance['Final Thought'] + "\n" + instance['output'],
            })
        except:
            err_count += 1
        else:
            train_examples.append({
                "tools": api_desc,
                "conversations": conv
            })

print("err_count:", err_count)
print("train_examples:", len(train_examples))
print("conversation distribution:", Counter([len(e["conversations"]) for e in train_examples]))

os.makedirs("formatted_data", exist_ok=True)

with open("formatted_data/tool_alpaca.jsonl", "w") as f:
    for e in train_examples:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")