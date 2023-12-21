#!/usr/bin/env python
# coding: utf-8

"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

part of this code is adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""
import math
import os
import sys
from dataclasses import dataclass, field
from glob import glob
from itertools import chain
from typing import Optional, List, Dict, Any, Mapping, Callable, Iterable, Union, Tuple, Sequence
from copy import deepcopy

import random

import numpy as np
import torch
from datasets import load_dataset, DatasetDict, IterableDatasetDict
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_int8_training
from sklearn.metrics import accuracy_score
from transformers import (
    AutoConfig,
    BloomForCausalLM,
    AutoModelForCausalLM,
    AutoModel,
    PreTrainedTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    BloomTokenizerFast,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.utils.versions import require_version

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cluster.dist_coordinator import DistCoordinator

index = int(os.environ["INDEX"]) or 0
logger.add(f"./pretraining_{index}.log")

# 修复多线程下 tiktoken 在datasets中的问题，https://github.com/huggingface/datasets/issues/5536#issuecomment-1682309347
import copyreg
import tiktoken
import functools


def pickle_Encoding(enc):
    return (
    functools.partial(tiktoken.core.Encoding, enc.name, pat_str=enc._pat_str, mergeable_ranks=enc._mergeable_ranks,
                      special_tokens=enc._special_tokens), ())


copyreg.pickle(tiktoken.core.Encoding, pickle_Encoding)

MODEL_CLASSES = {
    "bloom": (AutoConfig, BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoConfig, AutoModel, AutoTokenizer),
    "llama": (AutoConfig, LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    "xverse": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_type: str = field(
        default=None,
        metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys())}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    resize_token_embeddings: bool = field(default=False, metadata={"help": "Whether to resieze the token embeddings "
                                                                           "if it doesn't match tokenizer"})
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device to map model to. If `auto` is passed, the device will be selected automatically. "},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )

    def __post_init__(self):
        if self.model_type is None:
            raise ValueError(
                "You must specify a valid model_type to run training. Available model types are " + ", ".join(
                    MODEL_CLASSES.keys()))
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The train text data file folder."})
    validation_file_dir: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on text file folder."},
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
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=True, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")


@dataclass
class PeftArguments(TrainingArguments):
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)
    n_gpu: Optional[int] = field(default=1)


def accuracy(predictions, references, normalize=True, sample_weight=None):
    return {
        "accuracy": float(accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight))
    }


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics, we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return accuracy(predictions=preds, references=labels)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    try:
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    except ValueError:  # quick fix by simply take the first example
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))

    return batch


# class GroupTextsBuilder:
#     def __init__(self, max_seq_length):
#         self.max_seq_length = max_seq_length

#     def __call__(self, examples):
#         # Concatenate all texts.
#         firsts = {k: examples[k][0][0] for k in examples.keys()}
#         lasts = {k: examples[k][0][-1] for k in examples.keys()}
#         contents = {k: sum([vi[1:-1] for vi in v], []) for k, v in examples.items()}
#         total_length = len(contents[list(examples.keys())[0]])

#         content_length = self.max_seq_length - 2
#         if total_length >= content_length:
#             total_length = (total_length // content_length) * content_length
#         # Split by chunks of max_len.
#         result = {
#             k: [[firsts[k]] + t[i: i + content_length] + [lasts[k]] for i in range(0, total_length, content_length)] for
#             k, t in contents.items()}
#         return result


class SavePeftModelTrainer(Trainer):
    """
    Trainer for lora models
    """

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        os.makedirs(output_dir, exist_ok=True)
        #         torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


def save_model(output_dir, model, tokenizer, args):
    """Save the model and the tokenizer."""
    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


#     torch.save(args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def format_numel_str(numel: int) -> str:
    B = 1024 ** 3
    M = 1024 ** 2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {format_numel_str(trainable_params)} || all params: {format_numel_str(all_param)} || trainable%: {100 * trainable_params / all_param}"
    )


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def load_raw_datasets(data_args: DataTrainingArguments, model_args: DataTrainingArguments, coordinator: DistCoordinator,
                      seed=1024) -> IterableDatasetDict:
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir
            )
    #     elif data_args.dataset_from_disk is not None:
    #         raw_datasets = load_from_disk(data_args.dataset_from_disk)
    # #         raw_datasets = raw_datasets.rename_column("content", "text")
    #         raw_datasets = raw_datasets['train'].train_test_split(test_size=data_args.validation_split_percentage/100.0)
    #         raw_datasets['validation'] = raw_datasets['test']
    #         del raw_datasets['test']
    else:
        data_files = {}
        dataset_args = {
            #             "num_proc": data_args.preprocessing_num_workers
        }
        if data_args.train_file_dir is not None and os.path.exists(data_args.train_file_dir):
            train_data_files = glob(f'{data_args.train_file_dir}/**/train*.parquet', recursive=True) + \
                               glob(f'{data_args.train_file_dir}/**/train*.jsonl', recursive=True)
            # Train data files must be same type, e.g. all txt or all jsonl
            #             types = [f.split('.')[-1] for f in train_data_files]
            #             if len(set(types)) > 1:
            #                 raise ValueError(f"train files must be same type, e.g. all txt or all jsonl, but got {types}")
            train_data_files = sorted(list(set(train_data_files)))
            random.Random(seed).shuffle(train_data_files)
            #             logger.info(f"train files: {train_data_files}")
            data_files["train"] = train_data_files
        if data_args.validation_file_dir is not None and os.path.exists(data_args.validation_file_dir):
            eval_data_files = glob(f'{data_args.train_file_dir}/**/validation*.parquet', recursive=True) + \
                              glob(f'{data_args.train_file_dir}/**/validation*.jsonl', recursive=True)
            #             logger.info(f"eval files: {eval_data_files}")
            data_files["validation"] = eval_data_files
            # Train data files must be same type, e.g. all txt or all jsonl
            types = [f.split('.')[-1] for f in eval_data_files]
            if len(set(types)) > 1:
                raise ValueError(f"train files must be same type, e.g. all txt or all jsonl, but got {types}")
        extension = "text" if data_files["train"][0].endswith('txt') else "json" if data_files["train"][0].endswith(
            'jsonl') else 'parquet'

        if coordinator.is_master():
            logger.info(f"extension: '{extension}', data_files: {data_files}")

        if extension == "text":
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in data_files.keys():
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                streaming=True,
                **dataset_args,
            )

            raw_datasets = raw_datasets['train'].train_test_split(
                test_size=data_args.validation_split_percentage / 100.0)
            raw_datasets['validation'] = raw_datasets['test']
            del raw_datasets['test']
        else:
            if data_args.streaming:
                raw_datasets = load_dataset(
                    extension,
                    data_files=data_files,
                    cache_dir=model_args.cache_dir,
                    streaming=True,
                    **dataset_args,
                )
            else:
                raw_datasets = load_dataset(
                    extension,
                    data_files=data_files,
                    cache_dir=model_args.cache_dir,
                    num_proc=data_args.preprocessing_num_workers,
                    **dataset_args,
                )

    #     logger.info(f"Raw datasets: {raw_datasets}")

    def remove_null_text(example):
        if 'text' in example:
            return example['text'] is not None and isinstance(example['text'], str)
        else:
            return True

    raw_datasets = raw_datasets.filter(remove_null_text)
    if isinstance(raw_datasets, IterableDatasetDict):
        raw_datasets = raw_datasets.shuffle(seed=seed, buffer_size=10000)
    else:
        raw_datasets = raw_datasets.shuffle(seed=seed)
    return raw_datasets


import torch.nn.functional as F

IGNORE_INDEX = -100


def supervised_tokenize(
        data_point: Dict[str, str], tokenizer: PreTrainedTokenizer, ignore_index: int = None, max_length: int = 4096
) -> Dict[str, Union[int, str, List[int]]]:
    """
    A tokenization function to tokenize an original pretraining data point as following:
        {"source": "", "target": "Beijing, the capital of the People's Republic of China, ...", "category": "geography"}
    """
    assert tokenizer.add_bos_token is False and tokenizer.add_eos_token is False, (
        "Initially set `tokenizer.add_bos_token` and `tokenizer.add_eos_token` to False, "
        "add <bos> and <eos> manually later"
    )
    if ignore_index is None:
        ignore_index = IGNORE_INDEX

    source_text = data_point["source"]  # `str`
    target_text = data_point["target"]  # `str`
    is_null_source = len(source_text) == 0

    source_text = tokenizer.bos_token + source_text
    target_text += tokenizer.eos_token
    sequence_text = source_text + target_text

    tokenized = tokenizer([source_text, sequence_text])["input_ids"]
    sequence_input_ids = tokenized[1]
    sequence_labels = deepcopy(sequence_input_ids)

    source_length = len(tokenized[0])
    if not is_null_source:
        sequence_labels[:source_length] = [ignore_index for _ in range(source_length)]

    # sequence truncation.
    if len(sequence_input_ids) > max_length:
        sequence_input_ids = sequence_input_ids[:max_length]
        sequence_labels = sequence_labels[:max_length]

    return dict(
        input_ids=sequence_input_ids,
        labels=sequence_labels,
        seq_length=len(sequence_input_ids),
        seq_category=data_point["category"] if 'category' in data_point else "unknown",
    )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    Collate instances for supervised dataset.
    Each instance is a tokenized dictionary with fields
    `input_ids`(List[int]), `labels`(List[int]) and `sequence`(str).
    """

    tokenizer: PreTrainedTokenizer
    max_length: int = 4096
    ignore_index: int = IGNORE_INDEX

    def __call__(self, instances: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """

        Args:
            instances (`Sequence[Dict[str, List[int]]]`):
                Mini-batch samples, each sample is stored in an individual dictionary.

        Returns:
            (`Dict[str, torch.Tensor]`): Contains the following `torch.Tensor`:
                `input_ids`: `torch.Tensor` of shape (bsz, max_len);
                `attention_mask`: `torch.BoolTensor` of shape (bsz, max_len);
                `labels`: `torch.Tensor` of shape (bsz, max_len), which contains `IGNORE_INDEX`.
        """
        assert isinstance(self.tokenizer.pad_token_id, int) and self.tokenizer.pad_token_id >= 0, (
            f"`{self.tokenizer.__class__.__name__}.pad_token_id` must be a valid non-negative integer index value, "
            f"but now `{self.tokenizer.pad_token_id}`"
        )

        # `List[torch.Tensor]`
        batch_input_ids = [
            torch.LongTensor(instance["input_ids"][: self.max_length])
            if len(instance["input_ids"]) > self.max_length
            else torch.LongTensor(instance["input_ids"])
            for instance in instances
        ]
        batch_labels = [
            torch.LongTensor(instance["labels"][: self.max_length])
            if len(instance["labels"]) > self.max_length
            else torch.LongTensor(instance["labels"])
            for instance in instances
        ]

        if self.tokenizer.padding_side == "right":
            input_ids = torch.nn.utils.rnn.pad_sequence(
                sequences=batch_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )  # (bsz, max_len)
            labels = torch.nn.utils.rnn.pad_sequence(
                sequences=batch_labels,
                batch_first=True,
                padding_value=self.ignore_index,
            )  # (bsz, max_len)
            # pad to max
            to_pad = self.max_length - input_ids.size(1)
            input_ids = F.pad(input_ids, (0, to_pad), value=self.tokenizer.pad_token_id)
            labels = F.pad(labels, (0, to_pad), value=self.ignore_index)
        elif self.tokenizer.padding_side == "left":
            reversed_input_ids = [seq.flip(dims=(0,)) for seq in batch_input_ids]
            reversed_input_ids = torch.nn.utils.rnn.pad_sequence(
                sequences=reversed_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )  # (bsz, max_len)
            input_ids = torch.flip(reversed_input_ids, dims=(1,))  # (bsz, max_len)
            reversed_labels = [seq.flip(dims=(0,)) for seq in batch_labels]
            reversed_labels = torch.nn.utils.rnn.pad_sequence(
                sequences=reversed_labels,
                batch_first=True,
                padding_value=self.ignore_index,
            )  # (bsz, max_len)
            labels = torch.flip(reversed_labels, dims=(1,))  # (bsz, max_len)
        else:
            raise RuntimeError(
                f"`{self.tokenizer.__class__.__name__}.padding_side` can only be `left` or `right`, "
                f"but now `{self.tokenizer.padding_side}`"
            )

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)  # `torch.BoolTensor`, (bsz, max_len)

        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def print_tokens_sum(tokenized_datasets, data_args: DataTrainingArguments):
    def calculate_token_length(examples):
        input_ids = examples['input_ids']
        return {"token_length":
                    [len(t) for t in input_ids]
                }

    token_sum_datasets = tokenized_datasets.map(
        calculate_token_length,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        desc="Calculating tokens sum on dataset",
    )

    for split in token_sum_datasets:
        logger.info(f"split: '{split}', tokens sum: {sum(token_sum_datasets[split]['token_length'])}")


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    coordinator = DistCoordinator()

    if coordinator.is_master():
        logger.warning(f"Model args: {model_args}")
        logger.warning(f"Data args: {data_args}")
        logger.warning(f"Training args: {training_args}")
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load tokenizer
    if not model_args.model_type:
        raise ValueError("Please specify a model_type, e.g. llama, chatglm, bloom, etc.")
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]
    if coordinator.is_master():
        logger.info(
            f"config_class: '{config_class}', model_class: '{model_class}', tokenizer_class: '{tokenizer_class}'")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }

    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_args.model_name_or_path
    if coordinator.is_master():
        logger.info(f"tokenizer_kwargs: {tokenizer_kwargs}")
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

    # Preprocessing the datasets.
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        result["labels"] = result["input_ids"].copy()
        return result

    raw_datasets = load_raw_datasets(data_args, model_args, coordinator=coordinator, seed=training_args.seed)
    if coordinator.is_master():
        logger.info(f"Raw datasets: {raw_datasets}")

    # Preprocessing the datasets.
    if training_args.do_train:
        if isinstance(raw_datasets, IterableDatasetDict):
            column_names = list(next(iter(raw_datasets["train"])).keys())
        else:
            column_names = list(raw_datasets["train"].features)
    else:
        if isinstance(raw_datasets, IterableDatasetDict):
            column_names = list(next(iter(raw_datasets["validation"])).keys())
        else:
            column_names = list(raw_datasets["validation"].features)
    if coordinator.is_master():
        logger.info(f"column_names: {column_names}")

    with training_args.main_process_first(desc="Dataset tokenization and grouping"):
        is_pairwise_data = "source" in column_names and "target" in column_names
        if isinstance(raw_datasets, DatasetDict):
            if not is_pairwise_data:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                )
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    num_proc=data_args.preprocessing_num_workers,
                    batched=True,
                )
            else:
                if coordinator.is_master():
                    logger.warning("Tokenizing on source and target column")
                tokenized_datasets = raw_datasets.map(
                    supervised_tokenize,
                    fn_kwargs={"tokenizer": tokenizer, "max_length": block_size},
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                )
                #                 tokenized_datasets = tokenized_datasets.sort(column_names=["seq_category", "seq_length"], reverse=True, keep_in_memory=False)
                #                 tokenized_datasets = tokenized_datasets.remove_columns(column_names=["seq_category", "seq_length"])
                lm_datasets = tokenized_datasets

            if coordinator.is_master():
                logger.info(f"tokenized_datasets: {tokenized_datasets}")
                print_tokens_sum(tokenized_datasets, data_args)

                logger.info(f"lm_datasets: {lm_datasets}")
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    # Load model
    if model_args.model_type and model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1
        if ddp:
            model_args.device_map = {"": int(os.environ["LOCAL_RANK"]) or 0}

        config = config_class.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=model_args.trust_remote_code,
            cache_dir=model_args.cache_dir
        )
        if coordinator.is_master():
            logger.info(f"model_name_or_path: '{model_args.model_name_or_path}', torch_dtype: '{torch_dtype}', " +
                        f"trust_remote_code: '{model_args.trust_remote_code}', cache_dir: '{model_args.cache_dir}', " +
                        f"device_map: '{model_args.device_map}'")
            logger.info(f"config: {config}")
        try:
            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                torch_dtype=torch_dtype,
                load_in_8bit=model_args.load_in_8bit,
                device_map=model_args.device_map,
                trust_remote_code=model_args.trust_remote_code
            )
        except FileNotFoundError as ex:
            logger.warning("Load model from safetensors failed, try to load pytorch binary")
            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                torch_dtype=torch_dtype,
                load_in_8bit=model_args.load_in_8bit,
                device_map=model_args.device_map,
                trust_remote_code=model_args.trust_remote_code,
                use_safetensors=False
            )
        model_vocab_size = model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        if coordinator.is_master():
            logger.info(f"Vocab of the base model: {model_vocab_size}, Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_args.resize_token_embeddings and model_vocab_size != tokenzier_vocab_size:
            tokenizer.pad_token = tokenizer.unk_token
            logger.info(f"Resize model embeddings to fit tokenizer: {model_vocab_size}->{tokenzier_vocab_size}")
            model.resize_token_embeddings(tokenzier_vocab_size)
    else:
        raise ValueError(f"Error, model_name_or_path is None, Continue PT must be loaded from a pre-trained model")

    if training_args.use_peft:
        if training_args.peft_path is not None:
            logger.info(f"Peft from pre-trained model: {training_args.peft_path}")
            model = PeftModel.from_pretrained(model, training_args.peft_path, is_trainable=True)
        else:
            if coordinator.is_master():
                logger.info("Init new peft model")
            target_modules = training_args.target_modules.split(',') if training_args.target_modules else None
            if target_modules and 'all' in target_modules:
                target_modules = find_all_linear_names(model, int4=False, int8=model_args.load_in_8bit)
            modules_to_save = training_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=training_args.lora_rank,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                modules_to_save=modules_to_save)
            if coordinator.is_master():
                logger.info(f"peft_config: {peft_config}")
            model = get_peft_model(model, peft_config)

        if model_args.load_in_8bit:
            model = prepare_model_for_int8_training(model)
    #         model.print_trainable_parameters()
    #         print_trainable_parameters(model)
    else:
        #         logger.info("Full parameters training")
        model = model.float()
    #         print_trainable_parameters(model)

    ## print model summary
    if coordinator.is_master():
        logger.debug(f"Model: {model}")
        print_trainable_parameters(model)

    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets['train']
    #         max_train_samples = len(train_dataset)
    #         if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
    #             max_train_samples = min(len(train_dataset), data_args.max_train_samples)
    #             train_dataset = train_dataset.select(range(max_train_samples))
    #         logger.debug(f"Num train_samples: {len(train_dataset)}")
    #         logger.debug("Tokenized training example:")
    #         logger.debug(tokenizer.decode(next(iter(train_dataset))['input_ids']))

    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
    #         max_eval_samples = len(eval_dataset)
    #         if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
    #             max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
    #             eval_dataset = eval_dataset.select(range(max_eval_samples))
    #         logger.debug(f"Num eval_samples: {len(eval_dataset)}")
    #         logger.debug("Tokenized eval example:")
    #         logger.debug(tokenizer.decode(next(iter(eval_dataset))['input_ids']))

    # Initialize our Trainer
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True
    model.enable_input_require_grads()
    if not ddp and torch.cuda.device_count() > 1:
        # Keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        logger.info("Keeps Trainer from trying its own DataParallelism when more than 1 gpu is available")
        model.is_parallelizable = True
        model.model_parallel = True

    logger.debug(f"model.hf_device_map: {model.hf_device_map}")

    if not is_pairwise_data:
        data_collator = fault_tolerance_data_collator
    else:
        #         data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, max_length=block_size)
        data_collator = fault_tolerance_data_collator

    trainer = SavePeftModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
    )

    # Training
    if training_args.do_train:
        if coordinator.is_master():
            logger.info("*** Train ***")
        #         logger.debug(f"Train dataloader example: {next(iter(trainer.get_train_dataloader()))}")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        if coordinator.is_master():
            logger.debug(f"Training metrics: {metrics}")
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info(f"Saving model checkpoint to {training_args.output_dir}")
        save_model(training_args.output_dir, model, tokenizer, training_args)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = max_eval_samples
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        if coordinator.is_master():
            logger.debug(f"Eval metrics: {metrics}")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
