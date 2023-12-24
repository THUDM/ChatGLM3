"""
This script is a part of a larger project for generating text using large language models.
It includes functionalities for finding engine files, parsing arguments, setting up configurations for different models,
and executing the generation process with various settings.
This script particularly supports models like ChatGLM3-6B and its variants,
handling quantization, serialization, and runtime aspects.


Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Modifications made by Yuxuan.Zhang @ ZhipuAI on 2023-12-24.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications:

1. Removed input_file, tokenizer_type, and other parameters unrelated to dialogue. Set num_beams to 1.
2. Adapted single turn dialogue into ChatGLM3-6B template and implemented multi-turn conversations.

"""

import argparse
import json
import torch
import transformers

from pathlib import Path
from typing import List

import tensorrt_llm
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import (GenerationSession, ModelConfig, SamplingConfig)


def find_engines(dir: Path, model_name: str = "*", dtype: str = "*", tp_size: str = "*", rank: str = "*") -> List[Path]:
    """
    Searches for engine files matching a specified pattern within a directory.
    This is typically used to locate compiled model files for efficient execution on specific hardware.
    Parameters:
        - dir: The directory to search.
        - model_name, dtype, tp_size, rank:
        Pattern matching parameters to filter engine files by model name, data type,
        tensor parallel size, and rank respectively.
    Returns:
        - A list of Paths pointing to the engine files.
    """

    template = f"{model_name}_{dtype}_tp{tp_size}_rank{rank}.engine"
    return list(dir.glob(template))


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        choices=[
                            "chatglm3_6b",
                            "chatglm3_6b_base",
                            "chatglm3_6b_32k"
                        ],
                        default="chatglm3_6b",
                        help='the name of the model')
    parser.add_argument('--max_output_len', type=int, default=4096)
    parser.add_argument('--engine_dir', type=str, default=None)
    parser.add_argument('--tokenizer_dir', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--random_seed', type=int, default=2023)
    parser.add_argument('--streaming', default=True, action='store_true')
    args = parser.parse_args(args)

    return args


def main():
    """
    The main execution function of the script. It orchestrates the text generation process
    by performing several key steps:
        - Parses command-line arguments to configure model details, output specifications,
        and other user-defined parameters.
        - Loads the model configuration from a specified directory and prepares the environment for text generation
        based on the model and hardware specifics.
        - Sets up the generation session with the appropriate model, tokenizer, and runtime configurations.
        - Enters a loop to continuously accept user input, generate text based on the provided prompts, and output
        the model's responses.
        - Handles special commands such as 'stop' to end the conversation and 'clear' to reset the chat history.
        - Manages resources and ensures that the generated text is properly formatted and presented to the user.
    The function is designed to be the entry point of the script, invoking all necessary components and managing the
    flow of data and control throughout the execution.
    """

    args = parse_arguments()

    config_path = Path(args.engine_dir) / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    dtype = config['builder_config']['precision']
    max_output_len = min(config['builder_config']['max_output_len'], args.max_output_len)
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    remove_input_padding = config['builder_config']['remove_input_padding']
    tp_size = config['builder_config']['tensor_parallel']
    pp_size = config['builder_config']['pipeline_parallel']
    world_size = tp_size * pp_size

    assert world_size == tensorrt_llm.mpi_world_size(), f'Engine world size ({tp_size} * {pp_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

    max_output_len = min(max_output_len, args.max_output_len)
    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank, tp_size=world_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    serialize_path = find_engines(
        dir=Path(args.engine_dir),
        model_name=args.model_name,
        dtype=dtype,
        tp_size=world_size,
        rank=runtime_rank)[0]

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    model_config = ModelConfig(vocab_size=config['builder_config']['vocab_size'],
                               num_layers=config['builder_config']['num_layers'],
                               num_heads=config['builder_config']['num_heads'] // tp_size,
                               num_kv_heads=(config['builder_config']['num_kv_heads'] + tp_size - 1) // tp_size,
                               hidden_size=config['builder_config']['hidden_size'] // tp_size,
                               gpt_attention_plugin=use_gpt_attention_plugin,
                               remove_input_padding=config['builder_config']['remove_input_padding'],
                               model_name=args.model_name,
                               paged_kv_cache=config['builder_config']['paged_kv_cache'],
                               quant_mode=QuantMode(config['builder_config']['quant_mode']),
                               dtype=dtype)

    sampling_config = SamplingConfig(
        end_id=tokenizer.eos_token_id,
        pad_id=tokenizer.pad_token_id,
        num_beams=1,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    sampling_config.random_seed = args.random_seed

    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
        session = GenerationSession

    decoder = session(model_config, engine_buffer, runtime_mapping)

    history = []
    while True:
        input_text_with_history = ""
        max_input_len = config['builder_config']['max_input_len']
        input_text = input("用户: ")
        if input_text.lower() == 'stop':
            break

        if input_text.lower() == 'clear':
            history = []
            print("ChatGLM3-6B: 对话历史已清空")
            continue

        history.append(input_text)

        for idx, content in enumerate(history):
            if idx % 2 != 0:
                input_text_with_history += "{}\n".format(content)
            else:
                input_text_with_history += "<|user|>{}\n<|assistant|>".format(content)

        tokenized = tokenizer(
            input_text_with_history,
            return_tensors="pt",
            padding=True,
            return_length=True
        )

        input_ids = tokenized['input_ids'].int()
        input_lengths = tokenized['length'].int()
        max_input_len_real = torch.max(input_lengths)
        if max_input_len_real > max_input_len:
            input_ids = input_ids[:, :max_input_len]
            input_lengths = torch.where(input_lengths > max_input_len, max_input_len, input_lengths)
        else:
            max_input_len = max_input_len_real
        if remove_input_padding:
            input_ids_no_padding = (torch.zeros(1, torch.sum(input_lengths), dtype=torch.int32))

            lengths_acc = torch.cumsum(torch.cat([torch.IntTensor([0]), input_lengths]), dim=0)

            for i in range(len(input_ids)):
                input_ids_no_padding[0, lengths_acc[i]:lengths_acc[i + 1]] = torch.IntTensor(
                    input_ids[i, max_input_len - input_lengths[i]:max_input_len])

            input_ids = input_ids_no_padding

        elif use_gpt_attention_plugin:
            input_ids_padding_right = torch.zeros_like(input_ids) + sampling_config.end_id
            for i, sample in enumerate(input_ids):
                nPadding = 0
                for token in sample:
                    if token == sampling_config.pad_id:
                        nPadding += 1
                    else:
                        break
                input_ids_padding_right[i, :len(sample[nPadding:])] = sample[nPadding:]
            input_ids = input_ids_padding_right
        input_lengths = torch.tensor([input_ids.shape[-1]], dtype=torch.int32)
        decoder.setup(1, max_input_len, max_output_len, 1)
        output = decoder.decode(
            input_ids.contiguous().cuda(),
            input_lengths.contiguous().cuda(),
            sampling_config,
            output_sequence_lengths=True,
            return_dict=True,
            streaming=args.streaming
        )

        print("ChatGLM3-6B:", end="")
        generated_text = ""
        if args.streaming:
            for output_item in output:
                output_id = output_item["output_ids"]
                output_sequence_lengths = output_item["sequence_lengths"]
                output_id = output_id[0, 0, output_sequence_lengths[0, 0] - 1]
                output_word = tokenizer.convert_ids_to_tokens(int(output_id))
                output_word = output_word.replace("▁", " ")
                output_word = tokenizer.convert_tokens_to_string(output_word)
                print(output_word, end="", flush=True)
                generated_text += output_word
            print("\n")
        else:
            torch.cuda.synchronize()
            output_ids = output["output_ids"][0]
            output = output_ids[0, input_lengths.item():]
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            print(generated_text)

        history.append(generated_text)

    del decoder
    print(f"Good bye!")


if __name__ == '__main__':
    main()
