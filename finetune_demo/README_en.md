# ChatGLM3-6B finetune

This directory provides fine-tuning examples of the ChatGLM3-6B model, including full fine-tuning and P-Tuning v2. In
terms of format, it provides multiple rounds of dialogue fine-tuning samples and input and output format fine-tuning
samples.

If the model is downloaded locally, the `THUDM/chatglm3-6b` field in this article and the code should be replaced with
the corresponding address to load the model locally.

Running the example requires `python>=3.10`. In addition to the basic `torch` dependency, the example code also requires
dependencies to run.

**We provide [sample notebook](lora_finetune.ipynb) to demonstrate how to use our fine-tuning code. **

```bash
pip install -r requirements.txt
```

## Test hardware standards

We only provide single-machine multi-card/multi-machine multi-card running examples, so you will need at least one
machine with multiple GPUs. In the **default configuration file** in this warehouse, we record the usage of video
memory:

+ SFT full fine-tuning: evenly distributed among 4 graphics cards, each graphics card occupies `48346MiB` of video
  memory.
+ P-TuningV2 fine-tuning: 1 graphics card, occupying `18426MiB` memory.
+ LORA fine-tuning: 1 graphics card, occupying `14082MiB` memory.

> Please note that this result is for reference only, and the memory usage may be different for different parameters.
> Please make adjustments based on your hardware conditions.

## Multi-turn dialogue format

The multi-round dialogue fine-tuning example adopts the ChatGLM3 dialogue format convention and adds
different `loss_mask` to different characters to calculate `loss` for multiple rounds of responses in one pass.

For data files, the sample adopts the following format

If you only want to fine-tune your model's conversational capabilities, rather than its tool capabilities, you should
organize your data in the following format.

```json
[
  {
    "conversations": [
      {
        "role": "system",
        "content": "<system prompt text>"
      },
      {
        "role": "user",
        "content": "<user prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant response text>"
      },
      // ... Muti Turn
      {
        "role": "user",
        "content": "<user prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant response text>"
      }
    ]
  }
  // ...
]
```

**Please note that this method will affect the tool calling function of the model when there are many fine-tuning steps
**

If you wish to fine-tune your model's dialog and tool capabilities, you should organize your data in the following
format.

```json
[
  {
    "tools": [
      // available tools, format is not restricted
    ],
    "conversations": [
      {
        "role": "system",
        "content": "<system prompt text>"
      },
      {
        "role": "user",
        "content": "<user prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant thought to text>"
      },
      {
        "role": "tool",
        "name": "<name of the tool to be called",
        "parameters": {
          "<parameter_name>": "<parameter_value>"
        },
        "observation": "<observation>"
        // don't have to be string
      },
      {
        "role": "assistant",
        "content": "<assistant response to observation>"
      },
      // ... Muti Turn
      {
        "role": "user",
        "content": "<user prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant response text>"
      }
    ]
  }
  // ...
]
```

- There is no need to manually insert the system prompt about the tool description. The `tools` field will be used
  during preprocessing using `json.dumps(..., ensure_ascii=False)`
  After formatting, insert it as the first system prompt.

- Each role can be accompanied by a `loss` field of type `bool`, indicating whether the content predicted by this field
  participates in `loss`
  calculate. If there is no such field, the sample implementation does not calculate `loss` for `system` and `user` by
  default, but calculates `loss` for other roles.

- `tool` is not a native role in ChatGLM3. The `tool` here will be automatically converted into an `assistant` with tool
  call `metadata` during the preprocessing stage.
  role (default `loss` is calculated) and an `observation` role representing the tool return value (`loss` is not
  calculated).

- The fine-tuning task of `Code interpreter` has not been implemented yet.

- The `system` role is optional, but if the `system` role exists, it must appear in `user`
  Before the character, the `system` character can only appear once in a complete dialogue data (regardless of single
  round or multiple rounds of dialogue).

## Dataset format example

Here we take the AdvertiseGen data set as an example,
You can download it
from [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing)
Or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) download the AdvertiseGen data set.
Place the decompressed AdvertiseGen directory in the `data` directory and convert it into the following format data set
yourself.

> Please note that the verification set is added to the current fine-tuning code. Therefore, for a complete set of
> fine-tuning data sets, the training data set and the verification data set must be included, and the test data set
> does
> not need to be filled in. Or directly use the validation data set instead.

```
{"conversations": [{"role": "user", "content": "Type#skirt*skirt length#skirt"}, {"role": "assistant", "content": "This is versatile Fashionable fairy skirt, the overall design is very elegant and casual. Every girl can instantly turn into a fairy after wearing it. The material is very light and breathable, making it very comfortable to wear in summer."} ]}
```

## Configuration file

Fine-tuning configuration files are located in the `config` directory and include the following files:

1. `ds_zereo_2 / ds_zereo_3.json`: deepspeed configuration file.
2. `lora.yaml / ptuning.yaml / sft.yaml`: Configuration files for different models, including model parameters,
   optimizer parameters, training parameters, etc. Some important parameters are explained as follows:
    + data_config section
        + train_file: The file path of the training data set.
        + val_file: The file path of the verification data set.
        + test_file: The file path of the test data set.
        + num_proc: Number of processes used when loading data.
    + max_input_length: The maximum length of the input sequence.
    + max_output_length: The maximum length of the output sequence.
    + training_args section
        + output_dir: Directory for saving models and other outputs.
        + max_steps: The maximum number of steps for training.
        + per_device_train_batch_size: training batch size per device (e.g. GPU).
        + dataloader_num_workers: The number of worker threads used when loading data.
        + remove_unused_columns: Whether to remove unused columns in the data.
        + save_strategy: model saving strategy (for example, how many steps should be saved).
        + save_steps: How many steps should be taken to save the model.
        + log_level: log level (such as info).
        + logging_strategy: logging strategy.
        + logging_steps: How many steps to log.
        + per_device_eval_batch_size: Evaluation batch size per device.
        + evaluation_strategy: Evaluation strategy (e.g. how many steps should be evaluated).
        + eval_steps: How many steps to evaluate.
        + predict_with_generate: Whether to use generate mode for prediction.
    + generation_config section
        + max_new_tokens: The maximum number of new tokens generated.
    + peft_config section
        + peft_type: The parameter valid adjustment type used (e.g. LORA).
        + task_type: task type, here is the causal language model (CAUSAL_LM).
    + Lora parameters:
        + r: LoRA rank.
        + lora_alpha: Scaling factor for LoRA.
        + lora_dropout: dropout probability used in LoRA layer
    + P-TuningV2 parameters:
        + num_virtual_tokens: The number of virtual tokens.

## Start fine-tuning

Use the following code to execute **single machine multiple cards/multiple machines multiple cards** operation.

```angular2html
cd finetune_demo
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=8  finetune_hf.py  data/AdvertiseGen/  THUDM/chatglm3-6b  configs/lora.yaml 
```

Execute **Single machine single card** operation through the following code.

```angular2html
cd finetune_demo
python finetune_hf.py data/AdvertiseGen/ THUDM/chatglm3-6b configs/lora.yaml
```

## Fine-tuning from a checkpoint

If you train according to the above method, each fine-tuning will start from scratch. If you want to fine-tune from a
half-trained model, you can add a fourth parameter, which has two ways to pass in:

1. `yes`, automatically start training from the last saved Checkpoint
2. `XX`, breakpoint number, for example, `600` means training from Checkpoint number 600

For example, this is an example of continuing fine-tuning from the last saved point
```angular2html

cd finetune_demo
python finetune_hf.py  data/AdvertiseGen/  THUDM/chatglm3-6b  configs/lora.yaml yes
```

## Use the fine-tuned model

### Verify the fine-tuned model in inference_hf.py

You can use our fine-tuned model in `finetune_demo/inference_hf.py`, which can be easily tested with just one line of
code.

```angular2html
python inference_hf.py your_finetune_path --prompt your prompt
```

In this way, the answer you get is a fine-tuned answer.

### Use the fine-tuned model in other demos in this repos or external repos

You can use our `lora` and fully parameterized fine-tuned models in any demo, as follows:

1. Use the method of reading the model in `finetune_demo/inference_hf.py` to replace the method of reading the model in
   the demo.

> Please note that for LORA and P-TuningV2 we do not merge the trained models, but in `adapter_config.json`
> The fine-tuning path is recorded in . If your original model location changes, you should modify the path
> of `base_model_name_or_path` in `adapter_config.json`.


> Please note that we have only tested using NVIDIA Hopper (representative GPU: H100) and Ampère (representative GPU:
> A100) architecture and series of graphics cards. If you use a graphics card with another architecture, you may
> experience
> 1. Unknown training problem/Video memory usage is different from the above.
> 2. The architecture is too low and does not support certain features.
> 3. The problem of reasoning effect.
     > The above three situations are problems that the community has encountered before. Although the probability is
     extremely low, if you encounter the above problems, you can try to solve them in the community.

```python
def load_model_and_tokenizer(
        model_dir: Union[str, Path], trust_remote_code: bool = True
) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=trust_remote_code
    )
    return model, tokenizer
```

2. Read the fine-tuned model, please note that you should use the location of the fine-tuned model, for example, if your
   model location is `/path/to/finetune_adapter_model`
   , the original model address is `path/to/base_model`, then you should use `/path/to/finetune_adapter_model`
   as `model_dir`.
3. After completing the above operations, you can use the fine-tuned model normally, and other calling methods remain
   unchanged.

### hint

1. Before starting training, the fine-tuning code will print the preprocessing information of the first training data (
   it is commented by default and can be uncommented), which is displayed as

```log
Sanity
Check >> >> >> >> >> >> >
'[gMASK]': 64790 ->   -100
'sop': 64792 ->   -100
'<|system|>': 64794 ->   -100
'': 30910 ->   -100
'\n': 13 ->   -100
'Answer': 20115 ->   -100
'the': 267 ->   -100
'following': 1762 ->   -100
...
'know': 683 ->   -100
'the': 267 ->   -100
'response': 3010 ->   -100
'details': 3296 ->   -100
'.': 30930 ->   -100
'<|assistant|>': 64796 ->   -100
'': 30910 ->  30910
'\n': 13 ->     13
'I': 307 ->    307
'need': 720 ->    720
'to': 289 ->    289
'use': 792 ->    792
...
<< << << << << << < Sanity
Check
```

words, each line represents a detokenized string, token_id and target_id in turn. Among them, `target_id` is the index
of `token_id` in the model vocabulary, and `-100` means that
Token does not participate in `loss` calculation.

2. The function of `_prepare_model_for_training` is to iterate through all the trainable parameters of the model and
   ensure that their data type is `torch.float32`.
   This is necessary in some cases because mixed precision training or other operations may change the data type of the
   model parameters. This code is opened by default and can be commented, but if you use
   If there is a problem with `half` format training, you can switch back to this code, and the video memory may
   increase.
3. In our [Huggingface model code](https://huggingface.co/THUDM/chatglm3-6b/blob/main/modeling_chatglm.py), there is the
   following content:
    ```python
   if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_caches[index],
                    use_cache,
                    use_reentrant=False
                )
   ```
   This may cause the video memory to increase during training, so if you have insufficient video memory, you can try
   changing ``` use_reentrant``` to `True`.
4. The fine-tuned model can use any model acceleration framework that supports `peft` loading. Here, we do not provide a
   demo.
5. There are certain differences between the fine-tuning data set format of this warehouse and the API fine-tuning data
   set format.
    + The `messages` field in the ZhipuAI API fine-tuning data set is the `conversation` field in this warehouse.
    + The fine-tuning file in ZhipuAI API is `jsonl`. In this warehouse, you need to simply change the file name
      to `json`.

## Citation

```

@inproceedings{liu2022p,
title={P-tuning: Prompt tuning can be comparable to fine-tuning across scales and tasks},
author={Liu, Xiao and Ji, Kaixuan and Fu, Yicheng and Tam, Weng and Du, Zhengxiao and Yang, Zhilin and Tang, Jie},
booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short
Papers)},
pages={61--68},
year={2022}
}

@misc{tang2023toolalpaca,
title={ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases},
author={Qiaoyu Tang and Ziliang Deng and Hongyu Lin and Xianpei Han and Qiao Liang and Le Sun},
year={2023},
eprint={2306.05301},
archivePrefix={arXiv},
primaryClass={cs.CL}
}

```