# ChatGLM3-6B 微调示例

本目录提供 ChatGLM3-6B 模型的微调示例，包括全量微调和 P-Tuning v2。格式上，提供多轮对话微调样例和输入输出格式微调样例。

如果将模型下载到了本地，本文和代码中的 `THUDM/chatglm3-6b` 字段均应替换为相应地址以从本地加载模型。

运行示例需要 `python>=3.9`，除基础的 `torch` 依赖外，示例代码运行还需要依赖 

```bash
pip install transformers==4.30.2 accelerate sentencepiece astunparse deepspeed
```

## 多轮对话格式

多轮对话微调示例采用 ChatGLM3 对话格式约定，对不同角色添加不同 `loss_mask` 从而在一遍计算中为多轮回复计算 `loss`。

### 数据格式和预处理

对于数据文件，样例采用如下格式

如果您仅希望微调模型的对话能力，而非工具能力，您应该按照以下格式整理数据。

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

**请注意，这种方法在微调的step较多的情况下会影响到模型的工具调用功能**

如果您希望微调模型的对话和工具能力，您应该按照以下格式整理数据。

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

- 关于工具描述的 system prompt 无需手动插入，预处理时会将 `tools` 字段使用 `json.dumps(..., ensure_ascii=False)` 格式化后插入为首条 system prompt。

- 每种角色可以附带一个 `bool` 类型的 `loss` 字段，表示该字段所预测的内容是否参与 `loss` 计算。若没有该字段，样例实现中默认对 `system`, `user` 不计算 `loss`，其余角色则计算 `loss`。

- `tool` 并不是 ChatGLM3 中的原生角色，这里的 `tool` 在预处理阶段将被自动转化为一个具有工具调用 `metadata` 的 `assistant` 角色（默认计算 `loss`）和一个表示工具返回值的 `observation` 角色（不计算 `loss`）。

- 目前暂未实现 `Code interpreter`的微调任务。

- `system` 角色为可选角色，但若存在 `system` 角色，其必须出现在 `user` 角色之前，且一个完整的对话数据（无论单轮或者多轮对话）只能出现一次 `system` 角色。

作为示例，我们使用 ToolAlpaca 数据集来进行微调。首先，克隆 [ToolAlpaca 数据集](https://github.com/tangqiaoyu/ToolAlpaca)，并使用

```bash
./scripts/format_tool_alpaca.py --path "ToolAlpaca/data/train_data.json"
```

将数据集处理成上述格式。在这里，我们有意将工具处理成了了 `list[str]` 这样的自然语言形式，以观察模型在微调前后对工具定义的理解能力。

### 微调模型

以下脚本提供了微调模型的参考方式。

```bash
./scripts/finetune_ds_multiturn.sh  # 全量微调
./scripts/finetune_pt_multiturn.sh  # P-Tuning v2 微调
```

### 部署

我们更新了 ChatGLM3 的综合 Demo，使其可以部署微调后的模型 checkpoint。

对于全量微调，可以使用以下方式进行部署

```bash
cd ../composite_demo
MODEL_PATH="path to finetuned model checkpoint" TOKENIZER_PATH="THUDM/chatglm3-6b" streamlit run main.py
```

对于 P-Tuning v2 微调，可以使用以下方式进行部署

```bash
cd ../composite_demo
MODEL_PATH="THUDM/chatglm3-6b" PT_PATH="path to p-tuning checkpoint" streamlit run main.py
```

## 输入输出格式

对于输入-输出格式，样例采用如下输入格式

```json
[
  {
    "prompt": "<prompt text>",
    "response": "<response text>"
  }
  // ...
]
```

预处理时，不会拼接任何角色标识符。

作为示例，我们使用 AdvertiseGen 数据集来进行微调。从 [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing) 或者 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) 下载处理好的 AdvertiseGen 数据集，将解压后的 `AdvertiseGen` 目录放到本目录下。

```bash
./scripts/format_advertise_gen.py --path "AdvertiseGen/train.json"
```

来下载和将数据集处理成上述格式。

### 微调模型

以下脚本提供了微调模型的参考方式。

```bash
./scripts/finetune_ds.sh  # 全量微调
./scripts/finetune_pt.sh  # P-Tuning v2 微调
```

### 推理验证

对于输入输出格式的微调，可使用 `inference.py` 进行基本的推理验证。

```bash
python inference.py \
    --pt-checkpoint "path to p-tuning checkpoint" \
    --model THUDM/chatglm3-6b 
```

```bash
python inference.py \
    --tokenizer THUDM/chatglm3-6b \
    --model "path to finetuned model checkpoint" 
```

### 提示

1. 微调代码在开始训练前，会先打印首条训练数据的预处理信息，显示为

    ```log
    Sanity Check >>>>>>>>>>>>>
             '[gMASK]':  64790 ->   -100
                 'sop':  64792 ->   -100
          '<|system|>':  64794 ->   -100
                    '':  30910 ->   -100
                  '\n':     13 ->   -100
              'Answer':  20115 ->   -100
                 'the':    267 ->   -100
           'following':   1762 ->   -100
                      ...
                'know':    683 ->   -100
                 'the':    267 ->   -100
            'response':   3010 ->   -100
             'details':   3296 ->   -100
                   '.':  30930 ->   -100
       '<|assistant|>':  64796 ->   -100
                    '':  30910 ->  30910
                  '\n':     13 ->     13
                   'I':    307 ->    307
                'need':    720 ->    720
                  'to':    289 ->    289
                 'use':    792 ->    792
                      ...
    <<<<<<<<<<<<< Sanity Check
    ```

    字样，每行依次表示一个 detokenized string, token_id 和 target_id。可在日志中查看这部分的 `loss_mask` 是否符合预期。若不符合，可能需要调整代码或数据。

2. 参考显存用量

    - P-Tuning V2 `PRE_SEQ_LEN=128`, `DEV_BATCH_SIZE=1`, `GRAD_ACCUMULARION_STEPS=16`,  `MAX_SEQ_LEN=2048` 配置下约需要 21GB 显存。
    - 全量微调时，`./scripts/finetune_ds_multiturn.sh` 中的配置（`MAX_SEQ_LEN=2048`, `DEV_BATCH_SIZE=16`, `GRAD_ACCUMULARION_STEPS=1`）恰好用满 4 * 80GB 显存。

3. 若尝试后发现显存不足，可以考虑
    - 尝试降低 `DEV_BATCH_SIZE` 并提升 `GRAD_ACCUMULARION_STEPS`
    - 尝试添加 `--quantization_bit 8` 或 `--quantization_bit 4`。
        - `PRE_SEQ_LEN=128`, `DEV_BATCH_SIZE=1`, `GRAD_ACCUMULARION_STEPS=16`, `MAX_SEQ_LEN=1024` 配置下，`--quantization_bit 8` 约需 12GB 显存，`--quantization_bit 4` 约需 7.6GB 显存。

## 参考文献

```
@inproceedings{liu2022p,
  title={P-tuning: Prompt tuning can be comparable to fine-tuning across scales and tasks},
  author={Liu, Xiao and Ji, Kaixuan and Fu, Yicheng and Tam, Weng and Du, Zhengxiao and Yang, Zhilin and Tang, Jie},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
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
