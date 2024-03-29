import torch
import time
import argparse
import numpy as np

from ipex_llm.transformers import AutoModel
from modelscope import AutoTokenizer
from transformers import AutoTokenizer

# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://github.com/THUDM/ChatGLM3/blob/main/PROMPT.md
CHATGLM_V3_PROMPT_FORMAT = "<|user|>\n{prompt}\n<|assistant|>"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for ModelScope ChatGLM3 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="ZhipuAI/chatglm3-6b",
                        help='The ModelScope repo id for the ChatGLM3 model to be downloaded'
                             ', or the path to the ModelScope checkpoint folder')
    parser.add_argument('--prompt', type=str, default="AI是什么？",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    # It is important to set `model_hub='modelscope'`, otherwise model hub is default to be huggingface
    model = AutoModel.from_pretrained(model_path,
                                      load_in_4bit=True,
                                      trust_remote_code=True,
                                      model_hub='modelscope')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)

    # Generate predicted tokens
    with torch.inference_mode():
        prompt = CHATGLM_V3_PROMPT_FORMAT.format(prompt=args.prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        st = time.time()
        # if your selected model is capable of utilizing previous key/value attentions
        # to enhance decoding speed, but has `"use_cache": false` in its model config,
        # it is important to set `use_cache=True` explicitly in the `generate` function
        # to obtain optimal performance with IPEX-LLM INT4 optimizations
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)
        end = time.time()
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f'Inference time: {end - st} s')
        print('-' * 20, 'Prompt', '-' * 20)
        print(prompt)
        print('-' * 20, 'Output', '-' * 20)
        print(output_str)