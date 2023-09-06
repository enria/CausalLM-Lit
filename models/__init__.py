from copy import deepcopy
import importlib

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead

from .mixins import PeftModuleMixin

def load_model(config): 
    tokenizer = AutoTokenizer.from_pretrained(config.pretrain.path, trust_remote_code=True)
    tokenizer.padding_side="left"
    if "tokenizer" in config:
        if "pad_token" in config.tokenizer:
            tokenizer.pad_token = config.tokenizer.pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer.truncation_side='left'

    if "quant" in config:
        bnb_config = BitsAndBytesConfig(
            bnb_4bit_compute_dtype=torch.float16,
            **config.quant
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.pretrain.path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            use_auth_token=True
        )
    else:
        if "precision" in config:
            model = AutoModelForCausalLM.from_pretrained(
                config.pretrain.path,
                trust_remote_code=True,
                torch_dtype=eval(config.precision.dtype)
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.pretrain.path,
                trust_remote_code=True
            )
    
    if "config" in config:
        for k in config.config:
            setattr(model.config, k, config.config[k])
    
    if "peft" in config:
        peft_config = LoraConfig(
            **config.peft,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    return model, tokenizer

def load_reinforcement_model(config):

    tokenizer_path = config.pretrain.path
    if "tokenizer_path" in config.pretrain:
        tokenizer_path = config.pretrain.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if "tokenizer" in config:
        if "pad_token" in config.tokenizer:
            tokenizer.pad_token = config.tokenizer.pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if "quant" in config:
        bnb_config = BitsAndBytesConfig(
            bnb_4bit_compute_dtype=torch.float16,
            **config.quant )
        
    peft_config = None
    if "peft" in config:
        peft_config = LoraConfig(
            **config.peft,
            task_type="CAUSAL_LM")
        
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.pretrain.path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        is_trainable=True
    )
    model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.pretrain.path,
        peft_config=peft_config,
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    
    if "config" in config:
        for k in config.config:
            setattr(model.pretrained_model.config, k, config.config[k])
            setattr(model_ref.pretrained_model.config, k, config.config[k])

    return model, model_ref, tokenizer


def load_reward_model(config):
    module_name, class_name = config.reward.class_name.rsplit(".",maxsplit=2)
    module = importlib.import_module("."+module_name, package="models")
    class_obj = getattr(module, class_name)
    if "args" in config.reward:
        reward_model = class_obj(**config.reward.args)
    else:
        reward_model = class_obj()
    return reward_model