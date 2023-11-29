# ChatGLM3-6B-base 微调示例

本目录提供 ChatGLM3-6B-base 模型的微调示例，目前，仅包含了Lora微调。

如果将模型下载到了本地，本文和代码中的 `THUDM/chatglm3-6b-base` 字段均应替换为相应地址以从本地加载模型。

运行示例需要 `python>=3.10`，除基础的 `torch` 依赖外，示例代码运行还需要依赖 

```bash
pip install -r requirements.txt
```
## 多轮对话格式

`base`模型不具备对话能力，仅能够生成单轮回复。如果你希望使用多轮对话模型，使用`Chat`模型进行微调。


## 数据集要求

格式上，请使用`alpaca`数据集。

```bash

{"context": "hello", "target": "hi,I am ChatGLM3"}

```
其中，`context`是对话的上文，也就是模型的输入，`target`是对话的下文，也就是模型的输出。

## 微调模型

以下脚本提供了微调模型的参考方式。
```bash
./scripts/finetune_lora.sh  # 使用Lora微调
```
如果你使用的是windows系统，请将脚本改为`bat`文件执行。同时，如果你是单卡运行环境，也可以将脚本中中的这行代码
```
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py 
```
修改为
```
python finetune.py 
```
请注意，这样修改后，你需要手动设置`CUDA_VISIBLE_DEVICES`环境变量，以控制使用哪张显卡。

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
                    '':      0 ->   -100
                    '':      0 ->   -100 （有若干个）
    <<<<<<<<<<<<< Sanity Check
    ```
    字样，每行依次表示一个 detokenized string, token_id 和 target_id。可在日志中查看这部分的 `loss_mask` 是否符合预期。若不符合，可能需要调整代码或数据。
2. 参考显存用量
   - 按照官方脚本的默认参数运行，每一张显卡占用显存为 `23GB`。
3. 若尝试后发现显存不足，可以考虑
    - 尝试降低 `DEV_BATCH_SIZE` 并提升 `GRAD_ACCUMULARION_STEPS`
    - 尝试降低 `MAX_SEQ_LEN`，但是这可能会影响模型的性能
4. 数据量和 loss 衰减
    - 训练中 loss 在数个 step 中回升属于正常现象，但是如果连续不降，则需要检查数据集是否存在问题（覆盖范围过广，相关性太低等）。你也可以选择降低 `LEARNING_RATE` 或者提升 `WARMUP_STEPS`。
    - 通常来说，loss值在 `0.5` 以下时，就属于比较好的状态。
5. 训练数据和轮次
    - `Max Steps` 为训练执行的步骤，与数据集的数量无关，这个参数不等于`Epoch`。
    - `Batch Size` 为每个GPU的batch size，如果你使用了多卡训练，那么实际的batch size为 `Batch Size * GPU数量`。

## 注意事项
+ 基座模型不支持对话，工具，代码生成等能力，仅支持文本生成。如果你需要对话能力，请使用`Chat`模型和对应的微调框架。