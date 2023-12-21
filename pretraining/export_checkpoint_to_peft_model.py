#!/usr/bin/env python
# coding: utf-8

"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

part of this code is adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""
import math
import os
from dataclasses import dataclass, field
from glob import glob
from itertools import chain
from typing import Optional, List, Dict, Any, Mapping

import numpy as np
import torch
from datasets import load_dataset, DatasetDict, IterableDatasetDict, IterableDataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_int8_training
from transformers import (
    AutoConfig,
    BloomForCausalLM,
    AutoModelForCausalLM,
    AutoModel,
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

# In[ ]:


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
    pass


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
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
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


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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
    logger.info(f"config_class: '{config_class}', model_class: '{model_class}', tokenizer_class: '{tokenizer_class}'")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }

    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_args.model_name_or_path
    logger.info(f"tokenizer_kwargs: {tokenizer_kwargs}")
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

    # Preprocessing the datasets.
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    block_size = 1

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

    def empty():
        yield from ()

    raw_datasets = IterableDatasetDict({
        "train": IterableDataset.from_generator(empty),
        "validation": IterableDataset.from_generator(empty)
    })
    logger.info(f"Raw datasets: {raw_datasets}")

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
        logger.info(f"Vocab of the base model: {model_vocab_size}, Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_args.resize_token_embeddings and model_vocab_size != tokenzier_vocab_size:
            tokenizer.pad_token = tokenizer.unk_token
            logger.info(f"Resize model embeddings to fit tokenizer: {model_vocab_size}->{tokenzier_vocab_size}")
            model.resize_token_embeddings(tokenzier_vocab_size)
    else:
        raise ValueError(f"Error, model_name_or_path is None, Continue PT must be loaded from a pre-trained model")

    ## print model summary
    logger.debug(f"Model: {model}")

    if training_args.use_peft:
        if training_args.peft_path is not None:
            logger.info(f"Peft from pre-trained model: {training_args.peft_path}")
            model = PeftModel.from_pretrained(model, training_args.peft_path, is_trainable=True)
        else:
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
            logger.info(f"peft_config: {peft_config}")
            model = get_peft_model(model, peft_config)

        if model_args.load_in_8bit:
            model = prepare_model_for_int8_training(model)
        model.print_trainable_parameters()
        logger.debug(f"Peft Model: {model}")
    else:
        logger.info("Full parameters training")
        model = model.float()
        print_trainable_parameters(model)

    with training_args.main_process_first(desc="Dataset tokenization and grouping"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
        )
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
        )

    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets['train']

    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]

    # Initialize our Trainer
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True
    model.enable_input_require_grads()
    if not ddp and torch.cuda.device_count() > 1:
        # Keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = SavePeftModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
    )

    trainer._load_from_checkpoint(training_args.resume_from_checkpoint)
    trainer.save_state()
    logger.info(f"Saving model checkpoint to {training_args.output_dir}")
    save_model(training_args.output_dir, model, tokenizer, training_args)


if __name__ == "__main__":
    main()
