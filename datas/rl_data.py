import json
from os import path
from random import Random

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .seq_data import SequenceDM, SequenceDataset


class RLDataset(SequenceDataset):

    def tokenize_func(self, example):
        """单样本tokenize处理"""
        query = example['input']
        q_ids = self.tokenizer.encode(text=query, add_special_tokens=False)
        
        if len(q_ids)>self.max_length: # truncation=left
            if self.input_truncation_side=="left":
                q_ids = q_ids[-self.max_length:]
            else:
                q_ids = q_ids[:self.max_length]
        
        #TODO should be flexiable to most language model
        input_ids = q_ids
        
        return {'input_ids': input_ids}

    def collate_fn(self, batch_data):
        """根据batch最大长度做padding"""
        st_max_length = self.max_length * 2
        pad_token_id = self.tokenizer.pad_token_id

        len_list = [len(d['input_ids']) for d in batch_data]
        batch_max_len = max(len_list)
        input_ids, input_texts, origins = [], [], []
        for len_of_d, d in sorted(zip(len_list, batch_data), key=lambda x: -x[0]):
            pad_len = batch_max_len - len_of_d
            ids = [pad_token_id] * pad_len + d['input_ids']
            if batch_max_len > st_max_length:
                ids = ids[: st_max_length]
            input_ids.append(torch.LongTensor(ids))
            input_texts.append(d['input'])
            if "origin" in d:
                origins.append(d["origin"])
        input_ids = torch.stack(input_ids)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id), 
            'inputs': input_texts,
            "origins": origins
        }
        return data_dict

class RLDM(SequenceDM):
    def setup(self, stage: str):
        if stage in [pl.trainer.states.TrainerFn.FITTING]:
            train_path = path.join(self.data_dir, self.train_name)
            if path.exists(train_path):
                train_data = json.load(open(train_path))
                val_path = path.join(self.data_dir, self.val_name)
                if path.exists(val_path):
                    val_data = json.load(open(val_path))
                else:
                    shuffle = Random(1994).shuffle
                    shuffle(train_data)
                    pivot = int(len(train_data)*0.9)
                    train_data, val_data = train_data[:pivot], train_data[pivot:]
            else:
                raise FileNotFoundError

            train_data = [self.convert(x) for x in train_data]
            val_data= [self.convert(x) for x in val_data]

            self.train_data = RLDataset(datas = train_data, tokenizer = self.tokenizer, max_length=self.max_length, input_truncation_side = self.input_truncation_side)
            self.val_data = RLDataset(val_data, self.tokenizer, max_length=self.max_length, mode="val", input_truncation_side = self.input_truncation_side)
            print("train_length:", len(self.train_data))
            print("valid_length:", len(self.val_data))
            self.train_dl = DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.train_data.collate_fn)
        
        elif stage==pl.trainer.states.TrainerFn.TESTING:
            with open(path.join(self.data_dir, self.test_name)) as fin:
                test_data = json.load(fin)
                test_data = [self.convert(x) for x in test_data]
            self.test_data = RLDataset(test_data, self.tokenizer, max_length=self.max_length, mode="test", input_truncation_side = self.input_truncation_side)
            print("test_length:", len(self.test_data))

        elif stage==pl.trainer.states.TrainerFn.PREDICTING:
            with open(path.join(self.data_dir, self.predict_name)) as fin:
                predict_data = json.load(fin)
                predict_data = [self.convert(x) for x in predict_data]
            for item in predict_data:
                item["output"] = "need prediction"
            self.predict_data = RLDataset(predict_data, self.tokenizer, max_length=self.max_length, mode="test", input_truncation_side = self.input_truncation_side)
            print("precition_length:", len(self.predict_data))

