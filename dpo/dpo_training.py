#!/usr/bin/env python
# coding: utf-8


import os
import sys
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType
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
    TrainingArguments,
    BitsAndBytesConfig,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from trl import DPOTrainer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# index = int(os.environ["INDEX"]) or 0

# logger.add(f"./dpo-{index}.log")

# torch.distributed.init_process_group(backend="nccl")


# os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


MODEL_CLASSES = {
    "bloom": (AutoConfig, BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoConfig, AutoModel, AutoTokenizer),
    "llama": (AutoConfig, LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
}


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with DPO
    """
    # Model arguments
    model_type: str = field(
        default=None,
        metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys())}
    )
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The model checkpoint for weights initialization."}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The tokenizer for weights initialization."}
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load the model in 4bit mode or not."})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    # Do not touch this type annotation or it will stop working in CLI
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Enable deepspeed and pass the path to deepspeed json config file (e.g. `ds_config.json`) or an already"
                " loaded json file as a dict"
            )
        },
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})
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
    use_flash_attention_2: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use flash attention 2. You must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    # Dataset arguments
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The input jsonl data file folder."})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation jsonl file folder."}, )
    #     template_name: Optional[str] = field(default="vicuna", metadata={"help": "The prompt template name."})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "Train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "Eval batch size per device"})
    max_source_length: Optional[int] = field(default=256, metadata={"help": "Max length of prompt input text"})
    max_target_length: Optional[int] = field(default=256, metadata={"help": "Max length of output text"})
    min_target_length: Optional[int] = field(default=4, metadata={"help": "Min length of output text"})
    model_max_length: Optional[int] = field(default=None)
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
        default=4, metadata={"help": "The number of processes to use for the preprocessing."},
    )
    # Training arguments
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})
    target_modules: Optional[str] = field(default=None)
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=16.0)
    peft_path: Optional[str] = field(default=None)
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the validation set."})
    beta: Optional[float] = field(default=0.1, metadata={"help": "The beta parameter for DPO loss"})
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "Learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "The lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "The number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "The weight decay"})
    optim: Optional[str] = field(default="adamw_hf", metadata={"help": "The optimizer type"})
    fp16: Optional[bool] = field(default=True, metadata={"help": "Whether to use fp16"})
    bf16: Optional[bool] = field(default=False, metadata={"help": "Whether to use bf16"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use gradient checkpointing"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "The number of gradient accumulation steps"}
    )
    save_steps: Optional[int] = field(default=50, metadata={"help": "X steps to save the model"})
    eval_steps: Optional[int] = field(default=50, metadata={"help": "X steps to evaluate the model"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "X steps to log the model"})
    save_strategy: Optional[str] = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
                " `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to"
                " `metric_for_best_model` will always be retained in addition to the most recent ones. For example,"
                " for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be"
                " retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`,"
                " it is possible that two checkpoints are saved: the last one and the best one (if they are different)."
                " Default is unlimited checkpoints"
            )
        },
    )
    output_dir: Optional[str] = field(default="outputs-dpo", metadata={"help": "The output directory"})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    evaluation_strategy: Optional[str] = field(default="steps", metadata={"help": "Evaluation strategy"})
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={"help": "Remove unused columns from the dataset if `datasets.Dataset` is used"},
    )
    report_to: Optional[str] = field(default="tensorboard", metadata={"help": "Report to wandb or tensorboard"})

    def __post_init__(self):
        if self.model_type is None:
            raise ValueError("You must specify a valid model_type to run training.")
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")


def format_numel_str(numel: int, base: int = 1000) -> str:
    B = base ** 3
    M = base ** 2
    K = base
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
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def return_prompt_and_responses(examples) -> Dict[str, str]:
    """Load the paired dataset and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """

    for column in ["prompt", "chosen", "rejected"]:
        examples[column] = [t.strip() for t in examples[column]]

    # baichuan-2 模板, 这里的 chosen 和 rejected不用包含 EOS token
#     return {
# #         "prompt": ["Question: " + question.replace("\\n", "\n") + "\n\nAnswer: " for question in examples["prompt"]],
#         "prompt": ["<|im_start|>user\n" + question + "<|im_end|>\n<|im_start|>assistant\n" for question in examples["prompt"]],
#         "chosen": examples["chosen"],
#         "rejected": examples["rejected"],
#     }

    # Yi-34B-Chat 模板

    return {
        #         "prompt": ["Question: " + question.replace("\\n", "\n") + "\n\nAnswer: " for question in examples["prompt"]],
        "prompt": ["<|im_start|>user\n" + question + "<|im_end|>\n<|im_start|>assistant\n" for question in
                   examples["prompt"]],
        "chosen": examples["chosen"],
        "rejected": examples["rejected"],
    }


def plot_table(values):
    n_col = len(values[0])
    len_col = [max([len(str(row[i_col])) for row in values] + [6]) for i_col in range(n_col)]
    print("+" + "+".join(["".center(len_v, "-") for len_v in len_col]) + "+")
    for row in values:
        print("|" + "|".join([str(v).center(len_v, " ") for v, len_v in zip(row, len_col)]) + "|")
        print("+" + "+".join(["".center(len_v, "-") for len_v in len_col]) + "+")


def print_special_token(tokenizer, model):
    # print special token
    attrs = ["pad_token", "pad_token_id", "bos_token", "bos_token_id", "eos_token", "eos_token_id", "unk_token",
             "unk_token_id", "model_max_length", "max_position_embeddings"]
    tokenizer_attrs, model_attrs = list(), list()
    for attr in attrs:
        if hasattr(tokenizer, attr):
            tokenizer_attrs.append(getattr(tokenizer, attr))
        else:
            tokenizer_attrs.append("-")
        if hasattr(model.config, attr):
            model_attrs.append(getattr(model.config, attr))
        else:
            model_attrs.append("-")

    values = [[""] + attrs, ["tokenizer"] + tokenizer_attrs, ["model"] + model_attrs]

    plot_table(values)


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    try:
        torch.distributed.init_process_group(backend="nccl")
        from cluster.dist_coordinator import DistCoordinator
        coordinator = DistCoordinator()
    except Exception as ex:
        logger.warning(ex)

        #     import torch.distributed as dist
        class DistCoordinator(object):
            def is_master(self):
                return args.local_rank in [0, -1]

        coordinator = DistCoordinator()

    if coordinator.is_master():
        logger.info(f"Parse args: {args}")

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if args.model_type == 'bloom':
        args.use_fast_tokenizer = True
    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "use_fast": args.use_fast_tokenizer,
        "trust_remote_code": args.trust_remote_code,
    }
    tokenizer_name_or_path = args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = args.model_name_or_path
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
    if args.model_max_length is not None:
        tokenizer.model_max_length = args.model_max_length
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0  # set as the <unk> token

    # Get datasets
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.cache_dir,
            )
    else:
        data_files = {}
        if args.train_file_dir is not None and os.path.exists(args.train_file_dir):
            train_data_files = glob(f'{args.train_file_dir}/**/train*.json', recursive=True) + glob(
                f'{args.train_file_dir}/**/train*.jsonl', recursive=True)
            if coordinator.is_master():
                logger.info(f"train files: {', '.join(train_data_files)}")
            data_files["train"] = train_data_files
        if args.validation_file_dir is not None and os.path.exists(args.validation_file_dir):
            eval_data_files = glob(f'{args.validation_file_dir}/**/validation*.json', recursive=True) + glob(
                f'{args.validation_file_dir}/**/validation*.jsonl', recursive=True)
            if coordinator.is_master():
                logger.info(f"eval files: {', '.join(eval_data_files)}")
            data_files["validation"] = eval_data_files
        raw_datasets = load_dataset(
            'json',
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.cache_dir,
            )
    if coordinator.is_master():
        logger.info(f"Raw datasets: {raw_datasets}")

    # Preprocessing the datasets
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length
    full_max_length = max_source_length + max_target_length

    # Preprocess the dataset
    train_dataset = None
    max_train_samples = 0
    if args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets['train']
        max_train_samples = len(train_dataset)
        if args.max_train_samples is not None and args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        if coordinator.is_master():
            logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
        tokenized_dataset = train_dataset.shuffle().map(
            return_prompt_and_responses,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        train_dataset = tokenized_dataset.filter(
            lambda x: 0 < len(x['prompt'] + x['chosen']) <= full_max_length
                      and 0 < len(x['prompt'] + x['rejected']) <= full_max_length
        )
        if coordinator.is_master():
            logger.debug(f"Num train_samples: {len(train_dataset)}. "
                     + "First train example: "
                     + train_dataset[0]['prompt'] + train_dataset[0]['chosen'])

    eval_dataset = None
    max_eval_samples = 0
    if args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        max_eval_samples = len(eval_dataset)
        if args.max_eval_samples is not None and args.max_eval_samples > 0:
            max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        if coordinator.is_master():
            logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
        eval_dataset = eval_dataset.map(
            return_prompt_and_responses,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        eval_dataset = eval_dataset.filter(
            lambda x: 0 < len(x['prompt'] + x['chosen']) <= full_max_length
                      and 0 < len(x['prompt'] + x['rejected']) <= full_max_length
        )
        if coordinator.is_master():
            logger.debug(f"Num eval_samples: {len(eval_dataset)}. "
                     + "First eval example: "
                     + eval_dataset[0]['prompt'] + eval_dataset[0]['chosen'])

    # Load model
    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1
    if ddp:
        args.device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}

    if coordinator.is_master():
        logger.info(f"Device map: {args.device_map}")
    if args.qlora and is_deepspeed_zero3_enabled():
        logger.warning("ZeRO3 are both currently incompatible with QLoRA.")
    config = config_class.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        cache_dir=args.cache_dir
    )

    if args.model_max_length is not None:
        if "max_position_embeddings" in config.__dict__:
            config.max_position_embeddings = args.model_max_length

    if coordinator.is_master():
        logger.info(config)
    if args.load_in_4bit or args.load_in_8bit:
        logger.info(f"Quantizing model, load_in_4bit: {args.load_in_4bit}, load_in_8bit: {args.load_in_8bit}")
    if coordinator.is_master():
        logger.info("Loading train model")
    model = model_class.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="flash_attention_2" if args.use_flash_attention_2 else None,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        ) if args.qlora else None,
    )
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False

    if coordinator.is_master():
        logger.info("Loading reference model")
    ref_model = model_class.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="flash_attention_2" if args.use_flash_attention_2 else None,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        ) if args.qlora else None,
    )
    ref_model.config.end_token_id = tokenizer.eos_token_id
    ref_model.config.pad_token_id = ref_model.config.eos_token_id

    #     # Initialize our Trainer
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True

    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        save_safetensors=False,  # don't save safetensors
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
        report_to=args.report_to,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        optim=args.optim,
        bf16=args.bf16,
        fp16=args.fp16,
        remove_unused_columns=args.remove_unused_columns,
        #         deepspeed=args.deepspeed,
        local_rank=args.local_rank,
        run_name=f"dpo_{args.model_type}",
    )

    # Initialize DPO trainer
    peft_config = None
    if args.use_peft:

        target_modules = args.target_modules.split(',') if args.target_modules else None
        if target_modules and 'all' in target_modules:
            target_modules = find_all_linear_names(model, int4=args.load_in_4bit, int8=args.load_in_8bit)

        logger.info("Fine-tuning method: LoRA(PEFT)")
        logger.info(f"Peft target_modules: {target_modules}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    else:
        if coordinator.is_master():
            logger.info("Fine-tuning method: Full parameters training")
    trainer = DPOTrainer(
        model,
        ref_model=ref_model,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config if args.use_peft else None,
        max_prompt_length=args.max_source_length,
        max_length=full_max_length,
    )

    if is_deepspeed_zero3_enabled():
        trainer.ref_model = Accelerator().prepare(trainer.ref_model)

    if coordinator.is_master():
        print_special_token(trainer.tokenizer, trainer.model)
        print_trainable_parameters(trainer.model)

    # Training
    if args.do_train:
        if coordinator.is_master():
            logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        if coordinator.is_master():
            logger.debug(f"Training metrics: {metrics}")
            logger.info(f"Saving model checkpoint to {args.output_dir}")
            trainer.save_model(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            trainer.model.save_pretrained(args.output_dir,
                                          safe_serialization=False,  # don't save safetensors
                                          )

            # Evaluation
    if args.do_eval:
        if coordinator.is_master():
            logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if coordinator.is_master():
            logger.debug(f"Eval metrics: {metrics}")


if __name__ == "__main__":
    main()
