# coding=utf-8
import os
from typing import Any, Mapping, Optional
import time

import lightning.pytorch as pl
from trl import PPOTrainer, PPOConfig
import numpy as np
import torch

from models import load_reinforcement_model, load_reward_model
from utils import Timer


class ReinforcementLMModel(pl.LightningModule):
    def __init__(self, config):
        # 1. Init parameters
        super(ReinforcementLMModel, self).__init__()
        
        self.config=config

        self.batch_size = config.batch_size
        self.lr = config.model_config.optim.lr

        self.model, self.ref_model, self.tokenizer = load_reinforcement_model(config.model_config)
        self.reward_model = load_reward_model(config.data_config)

        ppo_config = PPOConfig(batch_size=config.batch_size,
                               mini_batch_size=16)

        self.ppo_trainer = PPOTrainer(config = ppo_config, 
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
            "top_k": 0.0,
            "top_p": 1.0,
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
        prefix = "model."
        if not self.model.is_peft_model:
            pretrained_model_state_dict = self.model.pretrained_model.state_dict(prefix=prefix+"pretrained_model.")
        else:
            # if it is a peft model, only save the v_head
            pretrained_model_state_dict = {}

        v_head_state_dict = self.model.v_head.state_dict()
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"{prefix}v_head.{k}"] = v
        return pretrained_model_state_dict
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return super().load_state_dict(state_dict, False)

    def training_step(self, batch, batch_idx):
        timing = {}

        self.ppo_trainer.config.batch_size = len(batch["inputs"])

        
        with Timer(timing, 'time/generate_response'):
            inputs = self.tokenizer(batch["inputs"], return_tensors="pt", padding=True, truncation=True).to(self.model.pretrained_model.device)
            outputs = self.model.generate(input_ids=inputs['input_ids'], 
                                        attention_mask=inputs['attention_mask'], 
                                        **self.gen_kwargs)
            prompt_tensors = [torch.masked_select(x, m.byte()) for x, m in zip(inputs['input_ids'], inputs['attention_mask'])]
            responses = outputs[:, inputs["input_ids"].shape[1]:]
            response_tensors = [torch.masked_select(x, x!=self.gen_kwargs["pad_token_id"]) for x in responses]
            response_tensors = [torch.cat((x, torch.LongTensor([self.tokenizer.eos_token_id]).to(x.device))) for x in response_tensors]
        
        with Timer(timing, "time/calcaulte_reward"):
            preds = [self.tokenizer.decode(output_ids, skip_special_tokens=True) 
                    for output_ids in response_tensors]
            rewards = self.reward_model(list(zip(batch['inputs'], preds)))

        with Timer(timing, "time/optimization"):
            stats = self.reinforcement_optimization_step(prompt_tensors,
                                                        response_tensors, 
                                                        rewards)

        logs={
            "env/reward_mean": np.mean(rewards),
            "env/reward_std": np.std(rewards),
            "env/reward_dist": rewards
        }
        logs.update(timing)
        logs.update(stats)
        logs["trainer/global_step"] = self.trainer.global_step
        
        # self.log_dict(logs)
        self.trainer.logger.experiment.log(logs)
    
    def reinforcement_optimization_step(self, query_tensors, response_tensors, rewards):
        self.trainer.fit_loop.epoch_loop.manual_optimization._on_before_step()
        stats = self.ppo_trainer.step(query_tensors,
                                      response_tensors, 
                                      rewards)
        self.trainer.fit_loop.epoch_loop.manual_optimization._on_after_step()
        return stats

    def validation_step(self, batch, batch_idx):
        if self.calculate_metric:
            raise NotImplementedError
        else:
            with Timer() as timer:
                inputs = self.tokenizer(batch["inputs"], return_tensors="pt", padding=True, truncation=True).to(self.model.pretrained_model.device)
                outputs = self.model.generate(input_ids=inputs['input_ids'], 
                                            attention_mask=inputs['attention_mask'], 
                                            **self.gen_kwargs)
                preds = [self.tokenizer.decode(output_ids, skip_special_tokens=True) 
                        for output_ids in outputs[:,inputs['input_ids'].shape[1]:]]
            rewards = self.reward_model(list(zip(batch['inputs'], preds)))
            self.validation_step_outputs.extend(rewards)
        
    def on_validation_epoch_end(self):
        if self.calculate_metric:
            pass
        else:
            reward = np.mean(self.validation_step_outputs)
            self.log("val_reward", reward)
        
        self.validation_step_outputs.clear()
    
    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end()
        
    def predict_step(self, batch, batch_idx):
        inputs = self.tokenizer(batch["inputs"], return_tensors="pt", padding=True, truncation=True).to(self.model.pretrained_model.device)
        outputs = self.model.generate(input_ids=inputs['input_ids'], 
                                    attention_mask=inputs['attention_mask'], 
                                    **self.gen_kwargs)
        
        pred = [self.tokenizer.decode(output_ids, skip_special_tokens=True) 
                for output_ids in outputs[0][:,inputs['input_ids'].shape[1]:]]
        self.validation_step_outputs.extend([(x, p) for x, p in zip(batch["origins"], pred)])
    
    def on_predict_epoch_end(self):
        from datas.seq_data import PredictionSaveMixin
        if  isinstance(self.trainer.datamodule, PredictionSaveMixin):
            self.trainer.datamodule.save_prediction(self.validation_step_outputs, os.path.join("output",self.config.prediction_output_name))
        self.validation_step_outputs.clear()