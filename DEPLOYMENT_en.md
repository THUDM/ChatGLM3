## Low-Cost Deployment

### Model Quantization

By default, the model is loaded with FP16 precision, running the above code requires about 13GB of VRAM. If your GPU's VRAM is limited, you can try loading the model quantitatively, as follows:

```python
model = AutoModel.from_pretrained("THUDM/chatglm3-6b",trust_remote_code=True).quantize(4).cuda()
```

Model quantization will bring some performance loss. Through testing, ChatGLM3-6B can still perform natural and smooth generation under 4-bit quantization.

### CPU Deployment

If you don't have GPU hardware, you can also run inference on the CPU, but the inference speed will be slower. The usage is as follows (requires about 32GB of memory):

```python
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).float()
```

### Mac Deployment

For Macs equipped with Apple Silicon or AMD GPUs, the MPS backend can be used to run ChatGLM3-6B on the GPU. Refer to Apple's [official instructions](https://developer.apple.com/metal/pytorch) to install PyTorch-Nightly (the correct version number should be 2.x.x.dev2023xxxx, not 2.x.x).

Currently, only [loading the model locally](README_en.md#load-model-locally) is supported on MacOS. Change the model loading in the code to load locally and use the MPS backend:

```python
model = AutoModel.from_pretrained("your local path", trust_remote_code=True).to('mps')
```

Loading the half-precision ChatGLM3-6B model requires about 13GB of memory. Machines with smaller memory (such as a 16GB memory MacBook Pro) will use virtual memory on the hard disk when there is insufficient free memory, resulting in a significant slowdown in inference speed.

### Multi-GPU Deployment

If you have multiple GPUs, but each GPU's VRAM size is not enough to accommodate the complete model, then the model can be split across multiple GPUs. First, install accelerate: `pip install accelerate`, and then load the model through the following methods:

```python
from utils import load_model_on_gpus

model = load_model_on_gpus("THUDM/chatglm3-6b", num_gpus=2)
```

This allows the model to be deployed on two GPUs for inference. You can change `num_gpus` to the number of GPUs you want to use. It is evenly split by default, but you can also pass the `device_map` parameter to specify it yourself.