#! /usr/bin/env python

import json
from collections import Counter
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)

args = parser.parse_args()

with open(args.path) as f:
    data = [json.loads(line) for line in f]

train_examples = [{
    "prompt": x['content'],
    "response": x['summary'],
} for x in data]

os.makedirs("formatted_data", exist_ok=True)

with open("formatted_data/advertise_gen.jsonl", "w") as f:
    for e in train_examples:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
