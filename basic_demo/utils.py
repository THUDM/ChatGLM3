"""
This utils script is designed to efficiently distribute the layers of a transformer-based language model across multiple GPUs.
It primarily addresses the challenge of ensuring that all components of the model are correctly allocated to the available GPUs,
which is essential for efficient parallel processing and preventing runtime errors, particularly in different operating systems.

The script contains two main functions:

1. auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
   - This function automatically configures a device map for the language model layers, given the number of GPUs available.
   The model is assumed to have 30 layers in total, including word embeddings, final layer normalization, and 28 transformer layers.
   - The function calculates how to distribute these 30 layers across the specified number of GPUs. It ensures that certain layers
   (word embeddings, final layer normalization, and the output layer) are always placed on the first GPU to
   avoid runtime errors that can occur due to mismatched device allocations in different operating systems.

2. load_model_on_gpus(checkpoint_path: Union[str, os.PathLike], num_gpus: int = 2, device_map: Optional[Dict[str, int]] = None, **kwargs) -> Module:
   - This function loads a transformer model from a specified checkpoint path onto the available GPUs.
   - If a custom device map is not provided, it uses the auto_configure_device_map function to create one.
   - The model is loaded in half precision (model.half()) for memory efficiency and dispatched across the GPUs as per the device map.

The script is adapted from the original source at https://github.com/THUDM/ChatGLM-6B, with modifications to support the
ChatGLM3 model and ensure compatibility across different operating systems, particularly addressing the device allocation issue in Linux.

Note: This script requires the 'transformers' and 'accelerate' libraries from Hugging Face for model loading and GPU dispatching.

Usage Example:
# Load a model onto 4 GPUs
model = load_model_on_gpus('path_to_checkpoint', num_gpus=4)
"""

import os
from typing import Dict, Union, Optional
from torch.nn import Module
from transformers import AutoModel


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus
    device_map = {
        'transformer.embedding.word_embeddings': 0,
        'transformer.encoder.final_layernorm': 0,
        'transformer.output_layer': 0,
        'transformer.rotary_pos_emb': 0,
        'lm_head': 0
    }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.encoder.layers.{i}'] = gpu_target
        used += 1

    return device_map

def load_model_on_gpus(checkpoint_path: Union[str, os.PathLike], num_gpus: int = 2,
                       device_map: Optional[Dict[str, int]] = None, **kwargs) -> Module:
    if num_gpus < 2 and device_map is None:
        model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half().cuda()
    else:
        from accelerate import dispatch_model

        model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half()

        if device_map is None:
            device_map = auto_configure_device_map(num_gpus)

        model = dispatch_model(model, device_map=device_map)

    return model
