import json
from os import path
from random import Random

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .seq_data import SequenceDM


class RLDataset(Dataset):
    def __init__(self, datas, tokenizer, max_length=1024, ignore_label_id=-100, mode="train") -> None:
        super().__init__()
        self.datas = datas
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_label_id = ignore_label_id
        self.mode = mode

    def tokenize_func(self, example):
        """单样本tokenize处理"""
        query = example['input']
        q_ids = self.tokenizer.encode(text=query, add_special_tokens=False)
        
        if len(q_ids)>self.max_length: # truncation=left
            q_ids = q_ids[self.max_length-1:]
        
        #TODO should be flexiable to most language model
        input_ids = q_ids
        
        return {'input_ids': input_ids}

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        item = self.tokenize_func(self.datas[index])
        item = dict(**item, **self.datas[index])
        return item

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
            if self.mode!="train" and "origin" in d:
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
    
    def train_dataloader(self):
        if hasattr(self,"train_dl"):
            return self.train_dl
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.train_data.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.val_batch_size, collate_fn=self.val_data.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.val_batch_size, collate_fn=self.test_data.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict_data, batch_size=self.val_batch_size, collate_fn=self.predict_data.collate_fn)

