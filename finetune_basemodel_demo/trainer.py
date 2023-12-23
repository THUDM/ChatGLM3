# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""
# coding=utf-8
import os
import torch
from transformers import Trainer
from transformers.modeling_utils import unwrap_model, PreTrainedModel
from transformers.utils import logging

logger = logging.get_logger(__name__)

WEIGHTS_NAME = "pytorch_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"


class LoRATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        model_to_save = unwrap_model(self.model)

        # Create a state_dict for saving, similar to the PrefixTrainer approach
        if isinstance(model_to_save, PreTrainedModel):
            state_dict = {k: v.to("cpu") for k, v in model_to_save.named_parameters() if v.requires_grad}
            # Using Hugging Face's save_pretrained instead of PyTorch's torch.save
            model_to_save.save_pretrained(output_dir, state_dict=state_dict, save_function=torch.save,
                                          safe_serialization=False)
        else:
            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))

        # Save tokenizer and training arguments as usual
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        print(self.args)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME, ))
