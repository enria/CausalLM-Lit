from copy import deepcopy
import importlib

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModel
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import AutoModelForCausalLMWithValueHead

from .mixins import PeftModuleMixin
from utils import chain_get


def load_model(config): 
    tokenizer = AutoTokenizer.from_pretrained(config.pretrain.path, trust_remote_code=True)
    tokenizer.padding_side="left"
    if "tokenizer" in config:
        for k in config.tokenizer:
            setattr(tokenizer, k, config.tokenizer[k])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side='left'

    fp16 = chain_get(config, ["quant", "fp16"], False)
    bf16 = chain_get(config, ["quant", "bf16"], True)

    compute_dtype = (torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float32))

    model_class = AutoModelForCausalLM
    if "model_class" in config:
        model_class = eval(config.model_class)

    if "quant" in config:
        bnb_config = BitsAndBytesConfig(
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16, # TODO 
            bnb_4bit_use_double_quant=True,
            **config.quant
        )
        model = model_class.from_pretrained(
            config.pretrain.path,
            quantization_config=bnb_config,
            load_in_4bit=chain_get(config, ["quant", "load_in_4bit"], False),
            load_in_8bit=chain_get(config, ["quant", "load_in_8bit"], False),
            torch_dtype=(torch.float32 if fp16 else (torch.bfloat16 if bf16 else torch.float32)),
            **dict(chain_get(config, ["pretrain", "args"], {}))
        )

        if compute_dtype == torch.float16 and chain_get(config, ["quant", "load_in_4bit"], False):
            major, minor = torch.cuda.get_device_capability()
            if major >= 8:
                print('='*80)
                print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
                print('='*80)
        
        # model.config.torch_dtype=(torch.float32 if fp16 else (torch.bfloat16 if bf16 else torch.float32))
        # setattr(model, 'model_parallel', True)
        # setattr(model, 'is_parallelizable', True)

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.gradient_checkpointing_enable()
    else:
        if "precision" in config:
            model = model_class.from_pretrained(
                config.pretrain.path,
                torch_dtype=eval(config.precision.dtype),
                **dict(chain_get(config, ["pretrain", "args"], {}))
            )
        else:
            model = model_class.from_pretrained(
                config.pretrain.path,
                **dict(chain_get(config, ["pretrain", "args"], {}))
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

        # from peft.tuners.lora import LoraLayer

        # for name, module in model.named_modules():
        #     if isinstance(module, LoraLayer):
        #         if bf16:
        #             module = module.to(torch.bfloat16)
        #     if 'norm' in name:
        #         module = module.to(torch.float32)
        #     if 'lm_head' in name or 'embed_tokens' in name:
        #         if hasattr(module, 'weight'):
        #             if bf16 and module.weight.dtype == torch.float32:
        #                 module = module.to(torch.bfloat16)

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
        is_trainable=True,
        load_in_4bit=chain_get(config, ["quant", "load_in_4bit"], False)
    )
    model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.pretrain.path,
        peft_config=peft_config,
        quantization_config=bnb_config,
        trust_remote_code=True,
        load_in_4bit=chain_get(config, ["quant", "load_in_4bit"], False)
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