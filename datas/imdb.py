import json
from os import path
from random import Random
from typing import Any

import lightning.pytorch as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
from datasets.download import DownloadMode
from tqdm import tqdm 

from .rl_data import RLDataset, RLDM
from .seq_data import SequenceDataset, SequenceDM

class LengthSampler:
    def __init__(self, min_value, max_value):
        self.values = list(range(min_value, max_value))
    def __call__(self):
        return np.random.choice(self.values)

class IMDbLMDM(SequenceDM):
    def __init__(self, **args):
        super().__init__(**args)

        self.input_size = LengthSampler(4, 8)
        self.output_size = LengthSampler(20, 50)

    def convert(self, item):
        sample =  {
            "input":  None,
            "output": None,
            "origin": item
        }

        tokens = item["text"].split(" ")
        input_tokens = tokens[:self.input_size()]
        output_tokens = tokens[len(input_tokens):len(input_tokens)+self.output_size()]
        sample["input"] = " ".join(input_tokens)
        sample["output"] = " ".join(output_tokens)
        return sample

    def setup(self, stage: str):
        if stage in [pl.trainer.states.TrainerFn.FITTING, pl.trainer.states.TrainerFn.TESTING]:

            train_data = load_dataset(self.data_dir, split='train')
            val_data = load_dataset(self.data_dir, split='test')

            train_data = [self.convert(x) for x in tqdm(train_data)]
            val_data= [self.convert(x) for x in val_data][:1000]

            self.train_data = SequenceDataset(train_data, self.tokenizer, max_length=self.max_length, ignore_input_loss=False)
            self.val_data = SequenceDataset(val_data, self.tokenizer, max_length=self.max_length, ignore_input_loss=False, mode="val")
            print("train_length:", len(self.train_data))
            print("valid_length:", len(self.val_data))
            self.train_dl = DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.train_data.collate_fn)

        else:
            raise NotImplemented

    def save_prediction(self, output, output_path):
        results = []
        for origin, pred in output:
            item = dict(**origin)
            item[self.predict_output_key] = pred
            results.append(item)
        with open(output_path, "w") as fout:
            json.dump(results, fout, indent=4, ensure_ascii=False)


class PositiveIMDbDM(RLDM):
    def __init__(self, data_dir = "imdb",  batch_size: int = 32, val_batch_size: int = -1,
                 tokenizer = None,  max_new_tokens = 35, max_length = 1024):
        pl.LightningDataModule.__init__(self)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        self.data_dir = data_dir

        self.input_size = LengthSampler(4, 8)
        self.output_size = LengthSampler(20, 50)

        if val_batch_size>0:
            self.val_batch_size = val_batch_size
        else:
            self.val_batch_size = batch_size

    def convert(self, item):
        sample =  {
            "input":  None,
            "origin": item
        }
        tokens = self.tokenizer.encode(item["text"])
        input_tokens = tokens[:self.input_size()]
        sample["input"] = self.tokenizer.decode(input_tokens)
        return sample

    def setup(self, stage: str):
        if stage == pl.trainer.states.TrainerFn.FITTING:

            train_data = load_dataset(self.data_dir, split='train')
            val_data = load_dataset(self.data_dir, split='test')

            train_data = [self.convert(x) for x in train_data if x["label"]==1]
            val_data= [self.convert(x) for x in val_data if x["label"]==1][:1000]

            self.train_data = RLDataset(train_data, self.tokenizer, max_length=self.max_length)
            self.val_data = RLDataset(val_data, self.tokenizer, max_length=self.max_length, mode="val")
            print("train_length:", len(self.train_data))
            print("valid_length:", len(self.val_data))
            self.train_dl = DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.train_data.collate_fn)

        elif stage == pl.trainer.states.TrainerFn.TESTING:
            test_data = load_dataset(self.data_dir, split='test')
            test_data= [self.convert(x) for x in test_data]

            self.test_data = RLDataset(test_data, self.tokenizer, max_length=self.max_length, mode="test")
            print("test_length:", len(self.test_data))

    def save_prediction(self, output, output_path):
        results = []
        for origin, pred in output:
            item = dict(**origin)
            item[self.predict_output_key] = pred
            results.append(item)
        with open(output_path, "w") as fout:
            json.dump(results, fout, indent=4, ensure_ascii=False)