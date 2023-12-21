#!/usr/bin/env python
# coding: utf-8

# 合并peft模型到原始模型中
# =====================
# **注意**: https://github.com/huggingface/peft.git@13e53fc 版本的peft没有了`merge_and_unload`函数，需要安装更高版本


import argparse

import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModel,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForSequenceClassification,
)
from loguru import logger

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "xverse": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default=None, type=str, required=True)
parser.add_argument('--base_model_name_or_path', default=None, required=True, type=str,
                    help="Base model name or path")
parser.add_argument('--tokenizer_path', default=None, type=str,
                    help="Please specify tokenization path.")
parser.add_argument('--peft_model_path', default=None, required=True, type=str,
                    help="Please specify LoRA model to be merged.")
parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
parser.add_argument('--output_dir', default='./merged', type=str)

args = parser.parse_args()

logger.info(args)

base_model_path = args.base_model_name_or_path
peft_model_path = args.peft_model_path
logger.info(f"Base model: {base_model_path}")
logger.info(f"LoRA model: {peft_model_path}")
peft_config = PeftConfig.from_pretrained(peft_model_path)
logger.info(f"Peft Config: {peft_config}")

model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
if peft_config.task_type == "SEQ_CLS":
    logger.info("Loading LoRA for sequence classification model")
    if args.model_type == "chatglm":
        raise ValueError("chatglm does not support sequence classification")
    try:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            #             device_map="auto",
        )
    except FileNotFoundError:
        logger.warning("Load model from safetensors failed, try to load pytorch binary")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            #             device_map="auto",
            use_safetensors=False
        )
else:
    logger.info("Loading LoRA for causal language model")
    try:
        base_model = model_class.from_pretrained(
            base_model_path,
            load_in_8bit=False,
            torch_dtype=torch.bfloat16,
            #             offload_folder = "./offload_dir",
            trust_remote_code=True,
            #             device_map="auto",
        )
    except FileNotFoundError:
        logger.warning("Load model from safetensors failed, try to load pytorch binary")
        base_model = model_class.from_pretrained(
            base_model_path,
            load_in_8bit=False,
            torch_dtype=torch.bfloat16,
            #             offload_folder = "./offload_dir",
            trust_remote_code=True,
            #             device_map="auto",
            use_safetensors=False
        )

if args.tokenizer_path:
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True)
else:
    tokenizer = tokenizer_class.from_pretrained(peft_model_path, trust_remote_code=True)
if args.resize_emb:
    base_model_token_size = base_model.get_input_embeddings().weight.size(0)
    if base_model_token_size != len(tokenizer):
        base_model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resize vocabulary size {base_model_token_size} to {len(tokenizer)}")

lora_model = PeftModel.from_pretrained(
    base_model,
    peft_model_path,
    #     device_map="auto",
    torch_dtype=torch.float16,
    #     offload_folder = "./offload_dir"
)
lora_model.eval()
logger.info(lora_model)

logger.info(f"Merging with merge_and_unload...")
base_model = lora_model.merge_and_unload()

logger.info("Saving to Hugging Face format...")
tokenizer.save_pretrained(args.output_dir)
base_model.save_pretrained(args.output_dir)
logger.info(f"Done! model saved to {args.output_dir}")
