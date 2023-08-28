import torch
from torch.utils.data import Dataset

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from .mixins import PredictionSaveMixin

class SequenceDataset(Dataset):
    def __init__(self, datas, tokenizer, max_length=1024, ignore_input_loss=True, ignore_label_id=-100, mode="train") -> None:
        super().__init__()
        self.datas = datas
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_label_id = ignore_label_id
        self.mode = mode

        self.ignore_input_loss = ignore_input_loss
        # self.datareader = DataReader()
        # self.samples, self.features = self.load_data()

    def tokenize_func(self, example):
        """单样本tokenize处理"""
        question = example['input']
        answer = example['output']
        q_ids = self.tokenizer.encode(text=question, add_special_tokens=False)
        a_ids = self.tokenizer.encode(text=answer, add_special_tokens=False)
        # if len(q_ids) > self.max_length - 2:  # 2 - gmask, bos
        #     q_ids = q_ids[: self.max_length - 2]
        # if len(a_ids) > self.max_length - 1:  # 1 - eos
        #     a_ids = a_ids[: self.max_length - 1]
        
        if len(q_ids) + len(a_ids)+1>self.max_length: # truncation=left
            q_ids = q_ids[self.max_length-1-len(q_ids):]
        
        question_length = len(q_ids)  # chatglm1 - gmask, bos, chatglm2 - gmask, sop

        # input_ids = self.tokenizer.build_inputs_with_special_tokens(q_ids, a_ids)
        #TODO should be flexiable to most language model
        if self.ignore_input_loss:
            input_ids = q_ids + a_ids + [self.tokenizer.eos_token_id]
            labels =    [self.ignore_label_id] * question_length + a_ids + [self.tokenizer.eos_token_id]
        else:
            input_ids = q_ids + a_ids + [self.tokenizer.eos_token_id]
            labels =    q_ids + a_ids + [self.tokenizer.eos_token_id]
        
        return {'input_ids': input_ids, 'labels': labels}

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        item = self.tokenize_func(self.datas[index])
        item = dict(**item, **self.datas[index])
        return item

    def collate_fn(self, batch_data):
        """根据batch最大长度做padding"""
        st_max_length = self.max_length*2
        pad_token_id = self.tokenizer.pad_token_id

        len_list = [len(d['input_ids']) for d in batch_data]
        tgt_len_list = [len(d['labels']) for d in batch_data]
        batch_max_len = max(len_list)
        batch_tgt_max_len = max(tgt_len_list)
        input_ids, labels, input_texts, output_texts, origins = [], [], [], [], []
        for len_of_d, t_len, d in sorted(zip(len_list, tgt_len_list, batch_data), key=lambda x: -x[0]):
            pad_len = batch_max_len - len_of_d
            tgt_pad_len = batch_tgt_max_len - t_len
            ids = d['input_ids'] + [pad_token_id] * pad_len
            label = d['labels'] + [self.ignore_label_id] * tgt_pad_len
            if batch_max_len > st_max_length:
                ids = ids[: st_max_length]
                label = label[: st_max_length]
            input_ids.append(torch.LongTensor(ids))
            labels.append(torch.LongTensor(label))
            input_texts.append(d['input'])
            output_texts.append(d['output'])
            if self.mode!="train" and "origin" in d:
                origins.append(d["origin"])
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id), 
            'inputs': input_texts,
            'outputs': output_texts,
            "origins": origins
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict


class SequenceDM(pl.LightningDataModule, PredictionSaveMixin):
    
    def train_dataloader(self):
        if hasattr(self,"train_dl"):
            return self.train_dl
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.train_data.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=self.val_data.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, collate_fn=self.test_data.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict_data, batch_size=self.batch_size, collate_fn=self.predict_data.collate_fn)
    