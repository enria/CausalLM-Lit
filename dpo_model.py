# coding=utf-8
import os
from typing import Any, Mapping, Optional
import time

import lightning.pytorch as pl
from trl import DPOTrainer, DPOConfig
import numpy as np
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from models import load_reinforcement_model, load_reward_model
from utils.utils import Timer, chain_get, torch_gc
import wandb


class ReinforcementLMModel(pl.LightningModule):
    def __init__(self, config):
        # 1. Init parameters
        super(ReinforcementLMModel, self).__init__()
        
        self.config=config

        self.batch_size = config.batch_size
        self.lr = config.model_config.optim.lr

        self.model, self.ref_model, self.tokenizer = load_reinforcement_model(config.model_config)
        self.model.is_peft_model=False # TODO 
        self.reward_model = load_reward_model(config.data_config)

        dpo_config = DPOConfig(learning_rate=config.model_config.optim.lr,
                               batch_size=config.batch_size,
                               mini_batch_size=config.mini_batch_size,
                               gradient_accumulation_steps=config.accumulate_grads)

        self.ppo_trainer = DPOTrainer(config = dpo_config,
                                      model=self.model, 
                                      ref_model= self.ref_model, 
                                      tokenizer= self.tokenizer)

        self.validation_step_outputs = []
        print("CausalLM model init: done.")
    
    def configure_optimizers(self):
        return None

    @property
    def calculate_metric(self):
        return bool(self.config.data_config.get("metrics") and 
           self.config.data_config["metrics"]["eval_calculate"])
    
    @property
    def gen_kwargs(self):
        return {
            "min_length":-1,
            "top_p": 0.5,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": self.trainer.datamodule.max_new_tokens
        }

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        return outputs
    
    def train(self, mode=True):
        return # dont change model state
    
    def state_dict(self):
        prefix = "model"
        v_head_state_dict = self.model.v_head.state_dict()
        if not self.is_peft_model:
            pretrained_model_state_dict = self.model.pretrained_model.state_dict(prefix=prefix+".pretrained_model.")
            for k, v in v_head_state_dict.items():
                pretrained_model_state_dict[f"{prefix}.v_head.{k}"] = v
        else:
            # if it is a peft model, need to save seperately
            pretrained_model_state_dict = {}
            peft_model_dict = get_peft_model_state_dict(self.model.pretrained_model)
            pretrained_model_state_dict = {
                f"{prefix}.pretrained_model.peft": peft_model_dict,
                f"{prefix}.v_head": v_head_state_dict
            }
        
        return pretrained_model_state_dict
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if not self.is_peft_model:
            return super().load_state_dict(state_dict, strict)
        else:
            prefix = "model"
            peft_weight = state_dict[f"{prefix}.pretrained_model.peft"]
            v_head = {
                "v_head": state_dict[f"{prefix}.v_head"]
            }
            self.model.load_state_dict(v_head, False)
            set_peft_model_state_dict(self.model.pretrained_model, peft_weight)

    @property
    def is_peft_model(self):
        return "peft" in self.config.model_config 

    def training_step(self, batch, batch_idx):
        timing = {}

        self.ppo_trainer.config.batch_size = len(batch["inputs"])
        
        with Timer(timing, 'time/generate_response'):
            query_tensors = [self.tokenizer(i, return_tensors="pt")["input_ids"].squeeze(dim=0) for i in batch["inputs"]]
            response = self.ppo_trainer.generate(query_tensors, 
                                                 batch_size = chain_get(self.config.data_config, ["ppo", "generate_batch_size"], 4), 
                                                 **self.gen_kwargs)
            response_tensors = [o[len(i):] for i,o in zip(query_tensors, response)]
        
        with Timer(timing, "time/calcaulte_reward"):
            preds = [self.tokenizer.decode(output_ids, skip_special_tokens=True) 
                    for output_ids in response_tensors]
            rewards, envs = self.reward_model(list(zip(batch['inputs'], preds, batch["origins"])))
        
        # torch_gc()

        with Timer(timing, "time/optimization"):
            stats = self.reinforcement_optimization_step(query_tensors, response_tensors, rewards)

        logs={
            "env/reward_mean": np.mean(rewards), "env/reward_std": np.std(rewards), "env/reward_dist": rewards
        }
        logs.update(envs)
        logs.update(timing)
        logs.update(stats)
        logs["trainer/global_step"] = self.trainer.global_step

        # table_rows = [list(r) for r in zip(batch["inputs"], preds, [r.item() for r in rewards])]
        # logs.update({"game_log": wandb.Table(columns=["query", "response", "reward"], rows=table_rows)})
        
        self.trainer.logger.experiment.log(logs)
    
    def reinforcement_optimization_step(self, query_tensors, response_tensors, rewards):
        self.trainer.fit_loop.epoch_loop.manual_optimization._on_before_step()
        if self.config.freeze:
            stats = {}
        else:
            stats = self.ppo_trainer.step(query_tensors, response_tensors,  rewards)
        self.trainer.fit_loop.epoch_loop.manual_optimization._on_after_step()
        return stats

    def validation_step(self, batch, batch_idx):
        padding_side_default = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(batch["inputs"], return_tensors="pt", padding=True, truncation=True).to(self.model.pretrained_model.device)
        with Timer():
            outputs = self.model.generate(input_ids=inputs['input_ids'], 
                                        attention_mask=inputs['attention_mask'],
                                        **self.gen_kwargs)
        preds = [self.tokenizer.decode(output_ids, skip_special_tokens=True) 
                for output_ids in outputs[:,inputs['input_ids'].shape[1]:]]
        self.tokenizer.padding_side = padding_side_default

        # torch_gc()

        if self.calculate_metric:
            self.validation_step_outputs.extend(list(zip(batch['inputs'], preds, batch["origins"])))
        else:
            rewards, envs = self.reward_model(list(zip(batch['inputs'], preds, batch["origins"])))
            self.validation_step_outputs.extend(rewards)
        
    def on_validation_epoch_end(self):
        if self.calculate_metric:
            metrics = self.trainer.datamodule.calculate_metrics(self.validation_step_outputs, self.reward_model)
            for key in metrics:
                self.log(f"{key}", metrics[key])
        else:
            reward = np.mean(self.validation_step_outputs)
            self.log("val_reward", reward)

        self.validation_step_outputs.clear()
    
    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end()
        
    def predict_step(self, batch, batch_idx):
        padding_side_default = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(batch["inputs"], return_tensors="pt", padding=True, truncation=True).to(self.model.pretrained_model.device)
        outputs = self.model.generate(input_ids=inputs['input_ids'], 
                                    attention_mask=inputs['attention_mask'],
                                    **self.gen_kwargs)
        preds = [self.tokenizer.decode(output_ids, skip_special_tokens=True) 
                for output_ids in outputs[:,inputs['input_ids'].shape[1]:]]
        self.tokenizer.padding_side = padding_side_default
        self.validation_step_outputs.extend([(x, p) for x, p in zip(batch["origins"], preds)])
    
    def on_predict_epoch_end(self):
        from datas.seq_data import PredictionSaveMixin
        if  isinstance(self.trainer.datamodule, PredictionSaveMixin):
            if not os.path.exists(self.config.predict_output_dir):
                os.makedirs(self.config.predict_output_dir)
            self.trainer.datamodule.save_prediction(self.validation_step_outputs, 
                                                    os.path.join(self.config.predict_output_dir, self.config.predict_output_name), 
                                                    self.reward_model)
        self.validation_step_outputs.clear()