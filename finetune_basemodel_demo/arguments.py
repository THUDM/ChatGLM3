from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    lora_checkpoint: str = field(
        default=None, metadata={"help": "Path to lora checkpoints"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "An optional parameter specifying the number of bits used for quantization. "
                "Quantization is a process that reduces the model size by limiting the number of "
                "bits that represent each weight in the model. A lower number of bits can reduce "
                "the model size and speed up inference, but might also decrease model accuracy. "
                "If not set (None), quantization is not applied."
            )
        },
    )

    lora_rank: Optional[int] = field(
        default=8,
        metadata={
            "help": (
                "balancing between complexity and model flexibility. A higher rank allows more "
                "complex adaptations but increases the number of parameters and computational cost."
            )
        },
    )
    lora_alpha: Optional[float] = field(
        default=32,
        metadata={
            "help": (
                "A higher value results in more significant adjustments, potentially improving adaptation to new tasks or data, "
                "but might also risk overfitting. A lower value makes smaller adjustments, possibly maintaining better generalization."
            )
        }, )

    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": (
                "during training to prevent the model from overly relying on specific patterns in the training data. "
                "Higher dropout rates can improve model generalization but may reduce learning efficiency."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )

    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )

    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    train_format: str = field(
        default=None, metadata={"help": "The format of the training data file (mulit-turn or input-output)"},
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        extension = self.train_file.split(".")[-1]
        assert extension in {"jsonl", "json"}, "`train_file` should be a jsonl or a json file."

        assert self.train_format in {"multi-turn", "input-output"}
