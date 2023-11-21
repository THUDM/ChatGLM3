from transformers import (
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from model.modeling_chatglm import ChatGLMForConditionalGeneration
import torch
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import datasets
import os
from argument import FinetuneArguments, CastOutputToFloat

tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

class GLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(**inputs).loss
    def save_model(self, output_dir=None, _internal_call=False):
        os.makedirs(output_dir, exist_ok=True)
        saved_params = {
            k: v.to("cuda") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "chatglm-lora.pt"))

def main():
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    model = ChatGLMForConditionalGeneration.from_pretrained(
        "model", load_in_8bit=False, trust_remote_code=True, device_map="auto"
    ).half()

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.lm_head = CastOutputToFloat(model.transformer.output_layer)
    model.config.use_cache = (
        False
    )
    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetune_args.lora_rank,
        target_modules=['query_key_value'],
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config).to("cuda")
    model.print_trainable_parameters()

    # load dataset
    dataset = datasets.load_from_disk(finetune_args.dataset_path)
    # start train
    trainer = GLMTrainer(
        model=model,
        data_collator=data_collator,
        train_dataset=dataset,
        args=training_args,
    )
    trainer.train()

if __name__ == "__main__":
    main()
