# -*- coding: utf-8 -*-
import os
import jieba
import dataclasses as dc
import functools
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Optional, Union
import numpy as np
import ruamel.yaml as yaml
import torch
import typer
from datasets import Dataset, DatasetDict, NamedSplit, Split, load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import (PeftConfig, PeftModelForCausalLM, get_peft_config,
                  get_peft_model)
from rouge_chinese import Rouge
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments,
    AutoConfig,
)
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq

from transformers import Seq2SeqTrainer as _Seq2SeqTrainer

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
# 定义一个 Typer 应用实例，用于创建命令行接口
app = typer.Typer(pretty_exceptions_show_locals=False)


class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):

    def __call__(self, features, return_tensors=None):
        output_ids = ([feature['output_ids'] for feature in features]
                      if 'output_ids' in features[0].keys() else None)
        if output_ids is not None:
            max_output_length = max(len(out) for out in output_ids)
            if self.pad_to_multiple_of is not None:
                max_output_length = (
                    (max_output_length + self.pad_to_multiple_of - 1) //
                    self.pad_to_multiple_of * self.pad_to_multiple_of)
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                    max_output_length - len(feature['output_ids']))
                if isinstance(feature['output_ids'], list):
                    feature['output_ids'] = feature['output_ids'] + remainder
                else:
                    feature['output_ids'] = np.concatenate(
                        [feature['output_ids'], remainder]).astype(np.int64)
        return super().__call__(features, return_tensors)


class Seq2SeqTrainer(_Seq2SeqTrainer):

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys=None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        if self.args.predict_with_generate:
            output_ids = inputs.pop('output_ids')
        input_ids = inputs['input_ids']
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)
        generated_tokens = generated_tokens[:, input_ids.size()[1]:]
        if self.args.predict_with_generate:
            labels = output_ids
        return loss, generated_tokens, labels

    # For P-Tuning a new save_model function is fine for the prefix_encoder model
    # but may cost problems for the whole model loading

    # def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
    #     if output_dir is None:
    #         output_dir = self.args.output_dir
    #     os.makedirs(output_dir, exist_ok=True)
    #     ptuning_params = {k: v for k, v in self.model.transformer.prefix_encoder.state_dict().items()}
    #
    #     torch.save(ptuning_params, os.path.join(output_dir, 'pytorch_model.bin'))
    #
    #     print(f"P-Tuning model weights saved in {output_dir}")
    #
    #     if self.tokenizer is not None:
    #         self.tokenizer.save_pretrained(output_dir)


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def _sanity_check(
    input_ids: Sequence[int],
    output_ids: Sequence[int],
    tokenizer: PreTrainedTokenizer,
):
    print('--> Sanity check')
    for in_id, out_id in zip(input_ids, output_ids):
        if in_id == 0:
            continue
        if in_id in tokenizer.tokenizer.index_special_tokens:
            in_text = tokenizer.tokenizer.index_special_tokens[in_id]
        else:
            in_text = tokenizer.decode([in_id])
        print(f'{repr(in_text):>20}: {in_id} -> {out_id}')


@functools.cache
def _get_yaml_parser() -> yaml.YAML:
    parser = yaml.YAML(typ='safe', pure=True)
    parser.indent(mapping=2, offset=2, sequence=4)
    parser.default_flow_style = False
    return parser


@dc.dataclass
class DataConfig(object):
    train_file: str
    val_file: Optional[str] = None
    test_file: Optional[str] = None

    num_proc: Optional[int] = None

    @property
    def data_format(self) -> str:
        return Path(self.train_file).suffix

    @property
    def data_files(self) -> dict[NamedSplit, str]:
        return {
            split: data_file
            for split, data_file in zip(
                [Split.TRAIN, Split.VALIDATION, Split.TEST],
                [self.train_file, self.val_file, self.test_file],
            ) if data_file is not None
        }


@dc.dataclass
class FinetuningConfig(object):
    data_config: DataConfig

    max_input_length: int
    max_output_length: int

    training_args: Seq2SeqTrainingArguments = dc.field(
        default_factory=lambda: Seq2SeqTrainingArguments(output_dir='./output'
                                                         ))
    peft_config: Optional[PeftConfig] = None

    def __post_init__(self):
        if not self.training_args.do_eval or self.data_config.val_file is None:
            # skips the evaluation stage when `do_eval` or `eval_file` is not provided
            self.training_args.do_eval = False
            self.training_args.evaluation_strategy = 'no'
            self.data_config.val_file = None
        else:
            self.training_args.per_device_eval_batch_size = (
                self.training_args.per_device_eval_batch_size
                or self.training_args.per_device_train_batch_size)

    @classmethod
    def from_dict(cls, **kwargs) -> 'FinetuningConfig':
        training_args = kwargs.get('training_args', None)
        if training_args is not None and not isinstance(
                training_args, Seq2SeqTrainingArguments):
            gen_config = training_args.get('generation_config')
            # TODO: a bit hacky
            if not isinstance(gen_config, GenerationConfig):
                training_args['generation_config'] = GenerationConfig(
                    **gen_config)
            kwargs['training_args'] = Seq2SeqTrainingArguments(**training_args)

        data_config = kwargs.get('data_config')
        if not isinstance(data_config, DataConfig):
            kwargs['data_config'] = DataConfig(**data_config)

        peft_config = kwargs.get('peft_config', None)
        if peft_config is not None and not isinstance(peft_config, PeftConfig):
            kwargs['peft_config'] = get_peft_config(peft_config)
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'FinetuningConfig':
        path = _resolve_path(path)
        kwargs = _get_yaml_parser().load(path)
        return cls.from_dict(**kwargs)


def _load_datasets(
    data_dir: Path,
    data_format: str,
    data_files: dict[NamedSplit, str],
    num_proc: Optional[int],
) -> DatasetDict:
    if data_format in ('.csv', '.json', '.jsonl'):
        dataset_dct = load_dataset(
            data_format[1:],
            data_dir=data_dir,
            data_files=data_files,
            num_proc=num_proc,
        )
    else:
        err_msg = f"Cannot load dataset in the '{data_format}' format."
        raise NotImplementedError(err_msg)

    return dataset_dct


class DataManager(object):

    def __init__(self, data_dir: str, data_config: DataConfig):
        self._num_proc = data_config.num_proc

        self._dataset_dct = _load_datasets(
            _resolve_path(data_dir),
            data_config.data_format,
            data_config.data_files,
            self._num_proc,
        )

    def _get_dataset(self, split: NamedSplit) -> Optional[Dataset]:
        return self._dataset_dct.get(split, None)

    def get_dataset(
        self,
        split: NamedSplit,
        process_fn: Callable[[dict[str, Any]], dict[str, Any]],
        batched: bool = True,
        remove_orig_columns: bool = True,
    ) -> Optional[Dataset]:
        orig_dataset = self._get_dataset(split)
        if orig_dataset is None:
            return

        if remove_orig_columns:
            remove_columns = orig_dataset.column_names
        else:
            remove_columns = None
        return orig_dataset.map(
            process_fn,
            batched=batched,
            remove_columns=remove_columns,
            num_proc=self._num_proc,
        )


def print_model_size(model: PreTrainedModel):
    print("--> Model")
    total_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print(f"\n--> model has {total_params / 1e6}M params\n")


def process_batch(
    batch: Mapping[str, Sequence],
    tokenizer: PreTrainedTokenizer,
    max_input_length: int,
    max_output_length: int,
) -> dict[str, list]:
    batched_tools = batch.get('tools', None)
    batched_conv = batch['conversations']
    batched_input_ids = []
    batched_labels = []

    if batched_tools is None:
        batched_tools = [None] * len(batched_conv)

    for tools, conv in zip(batched_tools, batched_conv):
        input_ids, loss_masks = [
            tokenizer.get_command('[gMASK]'),
            tokenizer.get_command('sop'),
        ], [False, False]

        if tools is not None:
            raise NotImplementedError()

        for message in conv:
            if message['role'] in ('system', 'user'):
                loss_mask_val = False
            else:
                loss_mask_val = True

            if message['role'] == 'tool':
                raise NotImplementedError()
            else:
                new_input_ids = tokenizer.build_single_message(
                    message['role'], '', message['content'])
                new_loss_masks = [loss_mask_val] * len(new_input_ids)

            input_ids += new_input_ids
            loss_masks += new_loss_masks

        input_ids.append(tokenizer.eos_token_id)
        loss_masks = [False, *loss_masks]
        labels = []
        for input_id, mask in zip(input_ids, loss_masks):
            if mask:
                labels.append(input_id)
            else:
                labels.append(-100)
        max_length = max_input_length + max_output_length + 1
        batched_input_ids.append(input_ids[:max_length])
        batched_labels.append(labels[:max_length])
    return {'input_ids': batched_input_ids, 'labels': batched_labels}


def process_batch_eval(
    batch: Mapping[str, Sequence],
    tokenizer: PreTrainedTokenizer,
    max_input_length: int,
    max_output_length: int,
) -> dict[str, list]:
    batched_tools = batch.get('tools', None)
    batched_conv = batch['conversations']
    batched_input_ids = []
    # To avoid computing loss, we do not provide the `labels` field in the input dictionary.
    batched_output_ids = []

    if batched_tools is None:
        batched_tools = [None] * len(batched_conv)

    for tools, conv in zip(batched_tools, batched_conv):
        input_ids = [
            tokenizer.get_command('[gMASK]'),
            tokenizer.get_command('sop'),
        ]

        if tools is not None:
            raise NotImplementedError()

        for message in conv:
            if len(input_ids) >= max_input_length:
                break
            if message['role'] == 'tool':
                raise NotImplementedError()
            else:
                new_input_ids = tokenizer.build_single_message(
                    message['role'], '', message['content'])
                if message['role'] == 'assistant':
                    output_prompt, output_ids = (
                        new_input_ids[:1],
                        new_input_ids[1:],
                    )
                    output_ids.append(tokenizer.eos_token_id)
                    batched_input_ids.append(input_ids[:max_input_length] +
                                             output_prompt[:1])
                    batched_output_ids.append(output_ids[:max_output_length])
                input_ids += new_input_ids
    return {'input_ids': batched_input_ids, 'output_ids': batched_output_ids}


# Not sure if this is necessary, can set it to half.
# If train with cpu, cast all params to fp32 instead of trainable ones.
def _prepare_model_for_training(model: nn.Module, use_cpu: bool):
    for param in model.parameters():
        if param.requires_grad or use_cpu:
            param.data = param.data.to(torch.float32)


def load_tokenizer_and_model(
    model_dir: str,
    peft_config: Optional[PeftConfig] = None,
) -> tuple[PreTrainedTokenizer, nn.Module]:
    """
    1. 加载预训练的分词器和模型，并根据提供的 peft_config 参数进行相应的微调。\n
    2. 如果 peft_config 包含 LORA 或 PREFIX_TUNING 微调配置，函数将根据这些配置调整模型。\n
    3 .如果 peft_config 为 None，则直接加载预训练模型。
    """
    # 加载预训练的分词器实例
    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              trust_remote_code=True)

    # 如果 peft_config 参数不为 None，说明需要进行 LORA 或 PREFIX_TUNING 微调
    if peft_config is not None:
        # 如果 peft_config 的 peft_type 名称是 "PREFIX_TUNING"
        if peft_config.peft_type.name == "PREFIX_TUNING":
            # 加载预训练的配置实例
            config = AutoConfig.from_pretrained(model_dir,
                                                trust_remote_code=True)
            # 设置前缀序列长度为 peft_config 中的 num_virtual_tokens 参数
            config.pre_seq_len = peft_config.num_virtual_tokens
            # 设置是否使用模型缓存，默认为 False
            config.use_cache = False
            # 使用预训练模型和调整后的配置实例创建模型实例
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                config=config,
            )
        # 如果 peft_config 的 peft_type 名称是 "LORA"
        elif peft_config.peft_type.name == "LORA":
            # 使用预训练模型目录和调整后的 peft_config 参数创建模型实例
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                empty_init=False,
                use_cache=False)
            # 使用 get_peft_model 函数对模型进行 LORA 微调
            # 这个函数的主要逻辑如下：
            """  检查模型类型： 确保输入的 model 是一个 PreTrainedModel 实例。
                - 如果 model 不是 PreTrainedModel 实例，函数将抛出一个 ValueError。
            检查 peft_config 类型：
                - 确保 peft_config 包含 LORA 配置。如果 peft_config 没有 LORA 配置，函数将抛出一个 ValueError。
            创建 LORA 适配器：
                - 如果 peft_config 包含 LORA 配置，函数将创建一个 PeftAdapter 实例。PeftAdapter 是用于 LORA 微调的适配器类。
            添加 LORA 适配器到模型：
                - 将创建的 PeftAdapter 实例添加到原始模型中。这将使模型能够使用 LORA 适配器进行微调。
            返回微调后的模型：
                - 返回微调后的模型，该模型现在包含 LORA 适配器，并准备进行微调。 """

            model = get_peft_model(model, peft_config)
            # 打印模型的可训练参数
            model.print_trainable_parameters()
    else:
        # 如果 peft_config 参数为 None，则直接使用预训练模型
        model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                     trust_remote_code=True,
                                                     empty_init=False,
                                                     use_cache=False)
    # 打印模型的尺寸信息
    print_model_size(model)
    # 返回分词器和模型实例
    return tokenizer, model


def compute_metrics(eval_preds: EvalPrediction,
                    tokenizer: PreTrainedTokenizer):
    batched_pred_ids, batched_label_ids = eval_preds

    metrics_dct = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-4': []}
    for pred_ids, label_ids in zip(batched_pred_ids, batched_label_ids):
        pred_txt = tokenizer.decode(pred_ids).strip()
        label_txt = tokenizer.decode(label_ids).strip()
        pred_tokens = list(jieba.cut(pred_txt))
        label_tokens = list(jieba.cut(label_txt))
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(pred_tokens),
                                  ' '.join(label_tokens))
        for k, v in scores[0].items():
            metrics_dct[k].append(round(v['f'] * 100, 4))
        metrics_dct['bleu-4'].append(
            sentence_bleu(
                [label_tokens],
                pred_tokens,
                smoothing_function=SmoothingFunction().method3,
            ))
    return {k: np.mean(v) for k, v in metrics_dct.items()}


# 使用 Typer 装饰器将 main 函数注册为应用的一个命令
@app.command()
def main(
    # 定义命令行参数 data_dir，用于指定数据目录
    data_dir: Annotated[str, typer.Argument(help='')],
    # 定义命令行参数 model_dir，用于指定模型目录
    model_dir: Annotated[
        str,
        typer.Argument(
            help=
            "A string that specifies the model id of a pretrained model configuration hosted on huggingface.co, or a path to a directory containing a model configuration file."
        ),
    ],
    # 定义命令行参数 config_file，用于指定配置文件路径
    config_file: Annotated[str, typer.Argument(help='')],

    # 定义命令行参数 auto_resume_from_checkpoint，用于指定是否自动恢复检查点
    auto_resume_from_checkpoint: str = typer.Argument(
        default='',
        help=
        'If entered as yes, automatically use the latest save checkpoint. If it is a numerical example 12 15, use the corresponding save checkpoint. If the input is no, restart training'
    ),
):
    # 从配置文件加载微调配置
    ft_config = FinetuningConfig.from_file(config_file)
    # 加载分词器和模型
    tokenizer, model = load_tokenizer_and_model(
        model_dir, peft_config=ft_config.peft_config)
    # 创建数据管理器
    data_manager = DataManager(data_dir, ft_config.data_config)

    # 获取训练数据集
    train_dataset = data_manager.get_dataset(
        Split.TRAIN,
        functools.partial(
            process_batch,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    print('train_dataset:', train_dataset)
    # 获取验证数据集
    val_dataset = data_manager.get_dataset(
        Split.VALIDATION,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if val_dataset is not None:
        print('val_dataset:', val_dataset)
    # 获取测试数据集
    test_dataset = data_manager.get_dataset(
        Split.TEST,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if test_dataset is not None:
        print('test_dataset:', test_dataset)

    # 检查编码数据集
    _sanity_check(train_dataset[0]["input_ids"], train_dataset[0]["labels"],
                  tokenizer)

    # 将模型转换为 fp32
    _prepare_model_for_training(model, ft_config.training_args.use_cpu)
    # 设置生成配置中的 pad_token_id 和 eos_token_id
    ft_config.training_args.generation_config.pad_token_id = (
        tokenizer.pad_token_id)
    ft_config.training_args.generation_config.eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.get_command('<|user|>'),
        tokenizer.get_command('<|observation|>'),
    ]
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    trainer = Seq2SeqTrainer(
        # 指定训练用的模型实例
        model=model,
        # 指定训练参数，包括输出目录、学习率、批量大小等
        args=ft_config.training_args,
        # 指定用于处理序列到序列任务数据的数据收集器
        data_collator=DataCollatorForSeq2Seq(
            # 指定分词器实例
            tokenizer=tokenizer,
            # 指定如何填充序列：'longest' 或 'max_length'
            padding='longest',
            # 指定如何返回张量：'pt' (PyTorch张量) 或 'tf' (TensorFlow张量)
            return_tensors='pt',
        ),
        # 指定训练数据集
        train_dataset=train_dataset,
        # 指定评估数据集
        eval_dataset=val_dataset.select(list(range(50))),
        # 如果使用的是 LORA 微调，不需要分词器
        tokenizer=tokenizer
        if ft_config.peft_config.peft_type != "LORA" else None,
        # 指定计算指标的函数
        compute_metrics=functools.partial(compute_metrics,
                                          tokenizer=tokenizer),
    )

    if auto_resume_from_checkpoint.upper(
    ) == "" or auto_resume_from_checkpoint is None:
        # 如果自动恢复检查点的参数为空或未指定，或者指定为否，则直接开始训练
        trainer.train()
    else:
        # 如果指定了自动恢复检查点，则从指定的检查点继续训练
        """
        如果指定为是，则检查模型输出目录中是否有可用的检查点，并尝试从最新的检查点恢复训练。
        如果指定的检查点编号存在，则使用该检查点继续训练。
        如果指定的检查点编号不存在或输入无效，则打印错误信息，并提示用户在模型输出目录中手动查找正确的检查点。 """
        output_dir = ft_config.training_args.output_dir
        dirlist = os.listdir(output_dir)
        checkpoint_sn = 0
        for checkpoint_str in dirlist:
            if checkpoint_str.find("eckpoint") > 0 and checkpoint_str.find(
                    "tmp") == -1:
                checkpoint = int(checkpoint_str.replace("checkpoint-", ""))
                if checkpoint > checkpoint_sn:
                    checkpoint_sn = checkpoint
        if auto_resume_from_checkpoint.upper() == "YES":
            if checkpoint_sn > 0:
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
                checkpoint_directory = os.path.join(
                    output_dir, "checkpoint-" + str(checkpoint_sn))
                print("resume checkpoint from  checkpoint-" +
                      str(checkpoint_sn))
                trainer.train(resume_from_checkpoint=checkpoint_directory)
            else:
                trainer.train()
        else:
            if auto_resume_from_checkpoint.isdigit():
                if int(auto_resume_from_checkpoint) > 0:
                    checkpoint_sn = int(auto_resume_from_checkpoint)
                    model.gradient_checkpointing_enable()
                    model.enable_input_require_grads()
                    checkpoint_directory = os.path.join(
                        output_dir, "checkpoint-" + str(checkpoint_sn))
                    print("resume checkpoint from  checkpoint-" +
                          str(checkpoint_sn))
                    trainer.train(resume_from_checkpoint=checkpoint_directory)
            else:
                print(
                    auto_resume_from_checkpoint,
                    "The specified checkpoint sn(" +
                    auto_resume_from_checkpoint +
                    ") has not been saved. Please search for the correct chkeckpoint in the model output directory"
                )

    # test stage
    """ 使用训练好的模型在 test_dataset 上进行预测。
    输出预测结果，通常包括生成文本和相应的评估指标，以评估模型的性能 """
    if test_dataset is not None:
        trainer.predict(test_dataset)


if __name__ == '__main__':
    app()
