# 使用NVIDIA TensorRT-LLM部署ChatGLM3

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main)是NVIDIA开发的高性能推理框架，您可以按照以下步骤来使用TensorRT-LLM部署ChatGLM3模型。

## 1. 安装TensorRT-LLM
#### 获取TensorRT-LLM代码：

```bash
# TensorRT-LLM 代码需要使用 git-lfs 拉取
apt-get update && apt-get -y install git git-lfs

git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM

# 本流程将使用 v0.7.0 Release 版本
git checkout tags/v0.7.0 -b release/0.7.0
git submodule update --init --recursive
git lfs install
git lfs pull
```

#### 构建docker镜像并安装TensorRT-LLM：
```bash
make -C docker release_build
```

#### 运行docker镜像：
```bash
make -C docker release_run
```

## 3. 为ChatGLM3模型构建TensorRT-LLM推理引擎：

#### 安装Python依赖：
```bash
cd ./examples/chatglm
pip install -r requirements.txt
apt-get update
apt-get install git-lfs
```
#### 从Huggingface下载ChatGLM3模型：
```
# 您可以选择具体想部署的模型下载
git clone https://huggingface.co/THUDM/chatglm3-6b      chatglm3_6b
git clone https://huggingface.co/THUDM/chatglm3-6b-base chatglm3_6b_base
git clone https://huggingface.co/THUDM/chatglm3-6b-32k  chatglm3_6b_32k
```

#### 使用build.py来构建推理引擎：
以下是一些使用build.py构建推理引擎的示例：
```bash
# 构建一个默认的精度为fp16的引擎
python3 build.py -m chatglm3_6b --output_dir trt_engines/chatglm3_6b/fp16/1-gpu

# 构建一个默认的精度为fp16的引擎，并打开FMHA功能（详见下文）
python3 build.py -m chatglm3_6b --enable_context_fmha --output_dir trt_engines/chatglm3_6b/fp16/1-gpu

# 构建一个w8a16的引擎
python3 build.py -m chatglm3_6b --use_weight_only --output_dir trt_engines/chatglm3_6b/weight_only/1-gpu

# 构建一个默认的精度为fp16的引擎，并支持使用两个GPU
python3 build.py -m chatglm3_6b --world_size 2 --output_dir trt_engines/chatglm3_6b/fp16/2-gpu

# 使用chatglm3_6b_base模型
python3 build.py -m chatglm3_6b_base --output_dir trt_engines/chatglm3_6b_base/fp16/1-gpu

# 使用chatglm3_6b-32k模型
python3 build.py -m chatglm3_6b_32k --output_dir trt_engines/chatglm3_6b-32k/fp16/1-gpu
```

#### 可配置的plugin参数

* 使用 `--use_gpt_attention_plugin <DataType>` 来配置 GPT Attention plugin (默认使用float16)。
* 使用 `--use_gemm_plugin <DataType>` 来配置 GEMM plugin (默认使用float16)。
* 使用 `--use_rmsnorm_plugin <DataType>` 来配置 RMS normolization plugin (默认使用float16)。

#### Fused Multi-Head Attention (FMHA)

* 使用 `--enable_context_fmha` 或 `--enable_context_fmha_fp32_acc` 参数来开启FMHA kernels, 可以获得更好的性能的同时降低显存开销。

* `--use_gpt_attention_plugin` 如果被设置为关闭的话将无法使用FMHA功能。

* `--enable_context_fmha` 将会使用FP16 accumulator, 可能会略微降低精度. 您也可以选择使用`--enable_context_fmha_fp32_acc` 来保护精度，但这会略微降低FMHA的性能提升。

#### Weight-Only 量化

* 使用 `--use_weight_only` 来开启 Weight-Only 量化, 这样可以加速推理并减少显存开销。

* 你还可以通过切换 `--weight_only_precision int8` 或者 `--weight_only_precision int4` 来选择具体是使用int8还是int4量化，默认为Int8。

#### In-flight Batching（须使用NVIDIA Triton进行推理）

* 使用 `--use_inflight_batching` 来开启 In-flight Batching，启用后，Paged KV Cache也会自动启用。

* Paged KV cache中block的数量可以用`--tokens_per_block` 来配置。

更多详细的功能和配置请参考：[TensorRT-LLM ChatGLM实现](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/chatglm)。

## 3. 使用TensorRT-LLM Python Runtime进行推理

#### 单机单卡的推理示例：

```bash
python3 ../run.py --input_text "What's new between ChatGLM3-6B and ChatGLM2-6B?" \
                  --max_output_len 50 \
                  --tokenizer_dir chatglm3_6b \
                  --engine_dir trt_engines/chatglm3_6b/fp16/1-gpu
```

#### 单机多卡的推理示例：

```bash
mpirun -n 2 \
    python ../run.py --input_text "What's new between ChatGLM3-6B and ChatGLM2-6B?" \
                     --max_output_len 50 \
                     --tokenizer_dir chatglm3_6b \
                     --engine_dir trt_engines/chatglm3_6b/fp16/1-gpu
```

* 如果您以root权限运行 `mpirun`，则可能需要添加 `--allow-run-as-root` 参数。

#### 运行summarize.py进行文章总结任务：

```bash
python3 ../summarize.py --test_trt_llm \
                        --hf_model_dir chatglm3_6b \
                        --engine_dir trt_engines/chatglm3_6b/fp16/1-gpu
```

#### 运行我们提供的对话demo:[tensorrt_llm_cli_demo.py](tensorrt_llm_cli_demo.py):

```bash
python3 tensorrt_llm_cli_demo.py --tokenizer_dir chatglm3_6b --engine_dir trt_engines/chatglm3_6b/fp16/1-gpu
```

运行结果展示：
```
用户: what is your name?
ChatGLM3-6B:Hello, I am an assistant named ChatGLM3-6B, and you can call me assistant. What can I help you with??

用户: what is new in ChatGLM3-6B compared with ChatGLM2-6B?
ChatGLM3-6B:ChatGLM3-6B is an improved version of ChatGLM2-6B. Compared with ChatGLM2-6B, ChatGLM3-6B has the following improvements:

1. Enhanced language understanding capabilities: ChatGLM3-6B's language model is based on the GLM3-6B model, which has been pre-trained on more diverse and large-scale data, resulting in better language understanding and generation capabilities.

2. Improved generation ability: ChatGLM3-6B has improved the generation ability compared to ChatGLM2-6B. With more training data and optimization algorithms, ChatGLM3-6B can generate more coherent and natural-looking text.

3. Enhanced adaptability to different dialogue scenarios: ChatGLM3-6B has been trained on more diverse dialogue data, including dialogue scenarios with different languages, cultures, and styles, making it more adaptable to different dialogue scenarios.

4. New features and functions: ChatGLM3-6B also has some new features and functions, such as support for multiple choice questions, sentiment analysis, and entity recognition.

In short, ChatGLM3-6B is more advanced and capable than ChatGLM2-6B, and can better meet the needs of users in various scenarios..
```

#### 性能测试：

您可以在[这里](https://github.com/NVIDIA/TensorRT-LLM/tree/main/benchmarks/python)查阅到如何测试 TensorRT-LLM 上运行 ChatGLM3 的性能。

## 4. 使用NVIDIA Triton部署在线推理服务器
使用 NVIDIA Triton 可以部署高性能，高拓展性，高稳定性的推理服务，并且可以开启In-flight Batching功能提升实际推理服务时的吞吐。详见[In-flight Batching Triton Backend](https://github.com/triton-inference-server/tensorrtllm_backend/tree/main/inflight_batcher_llm)。