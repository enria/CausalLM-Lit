# coding=utf-8
import os
from typing import Any, Mapping, Optional

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers import get_linear_schedule_with_warmup 
from models import load_model, PeftModuleMixin

class CausalLMModel(PeftModuleMixin, pl.LightningModule):
    def __init__(self, config):
        # 1. Init parameters
        super(CausalLMModel, self).__init__()
        
        self.config=config

        self.batch_size = config.batch_size
        self.lr = config.model_config.optim.lr

        self.model, self.tokenizer = load_model(config.model_config)
        
        self.model.train()
        print('num tokens:', len(self.tokenizer))

        self.validation_step_outputs = []

        print("CausalLM model init: done.")
    
    def configure_optimizers(self):
        import torch.optim as optim
        arg_list = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim.AdamW(arg_list, lr=self.lr)
        if self.config.disable_linear_schedule:
            return optimizer

        total_steps = int(len(self.trainer.datamodule.train_dataloader()) // self.config.accumulate_grads ) * self.config.max_epochs # accumulate_grads
        warmup_step =  int(total_steps * self.config.warmup_rate)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=total_steps)        

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 1, 'strict': True, 'monitor': None}]

    @property
    def calculate_metric(self):
        return bool(self.config.data_config.get("metrics") and 
           self.config.data_config["metrics"]["eval_calculate"])
    
    @property
    def is_peft_mode(self):
        return "peft" in self.config.model_config

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])

        self.log('train_loss',outputs.loss.item(), prog_bar=True, on_step=True)
        return {'loss': outputs.loss}

    def validation_step(self, batch, batch_idx):
        if self.calculate_metric:
            inputs = self.tokenizer(batch["inputs"], return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            outputs = self.model.generate(input_ids=inputs['input_ids'], 
                                        attention_mask=inputs['attention_mask'], 
                                        num_beams=self.config.num_beams, 
                                        max_new_tokens=self.trainer.datamodule.max_new_tokens, 
                                        # use_cache=True, # chatglm2 一定要使用cache
                                        return_dict_in_generate=True,
                                        output_hidden_states=True, 
                                        output_scores=True,
                                        pad_token_id=self.tokenizer.eos_token_id)
            
            preds = [self.tokenizer.decode(output_ids, skip_special_tokens=True) 
                    for output_ids in outputs[0][:,inputs['input_ids'].shape[1]:]]
            
            if self.config.data_config.metrics.get("origin"):
                batch_outputs = list(zip(batch['outputs'], preds, batch["origins"]))
            else:
                batch_outputs = list(zip(batch['outputs'], preds))
            self.validation_step_outputs.extend(batch_outputs)
        else:
            outputs = self(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            self.validation_step_outputs.append(outputs.loss.item())
        
    def on_validation_epoch_end(self):
        if self.calculate_metric:
            metrics = self.trainer.datamodule.calculate_metrics(self.validation_step_outputs)
            for key in metrics:
                self.log(f"{key}", metrics[key], sync_dist=True)
        else:
            loss = sum(self.validation_step_outputs)/len(self.validation_step_outputs)
            self.log("val_loss", loss)
        
        self.validation_step_outputs.clear()
    
    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end()
        
    def predict_step(self, batch, batch_idx) :
        inputs = self.tokenizer(batch["inputs"], return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        outputs = self.model.generate(input_ids=inputs['input_ids'], 
                                    attention_mask=inputs['attention_mask'], 
                                    num_beams=self.config.num_beams, 
                                    max_new_tokens=self.trainer.datamodule.max_new_tokens, 
                                    use_cache=True,
                                    return_dict_in_generate=True,
                                    output_hidden_states=True, 
                                    output_scores=True,
                                    pad_token_id=self.tokenizer.eos_token_id)
        
        pred = [self.tokenizer.decode(output_ids, skip_special_tokens=True) 
                for output_ids in outputs[0][:,inputs['input_ids'].shape[1]:]]
        self.validation_step_outputs.extend([(x, p) for x, p in zip(batch["origins"], pred)])
    
    def on_predict_epoch_end(self):
        from datas.seq_data import PredictionSaveMixin
        if  isinstance(self.trainer.datamodule, PredictionSaveMixin):
            self.trainer.datamodule.save_prediction(self.validation_step_outputs, 
                                                    os.path.join(self.config.predict_output_dir,
                                                                 self.config.predict_output_name))
        self.validation_step_outputs.clear()