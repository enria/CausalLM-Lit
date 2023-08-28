import json
from os import path
from random import Random

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from .seq_data import SequenceDataset, SequenceDM

class InstructionDM(SequenceDM):
    def __init__(self, data_dir = "",  batch_size: int = 32, 
                 tokenizer = None,  max_new_tokens = 128, max_length = 1024,
                 train_name="train.json", val_name="val.json", predict_name="predict.json",
                 predict_output_key="output"):
        pl.LightningDataModule.__init__(self)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        self.data_dir = data_dir
        self.train_name = train_name
        self.val_name = val_name
        self.predict_name = predict_name

        self.predict_output_key = predict_output_key

    def generate_prompt(self, example):
        if example["input"]:
            return (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
            )
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
        )
    
    def convert(self, item):
        return {
            "input":  self.generate_prompt(item),
            "output": item["output"]
        }

    def setup(self, stage: str):
        if stage in [pl.trainer.states.TrainerFn.FITTING, pl.trainer.states.TrainerFn.TESTING]:
            train_path = path.join(self.data_dir, self.train_name)
            if path.exists(train_path):
                train_data = json.load(open(train_path))
                val_path = path.join(self.data_dir, self.val_name)
                if path.exists(val_path):
                    val_data = json.load(open(val_path))
                else:
                    shuffle = Random(42).shuffle
                    shuffle(train_data)
                    pivot = int(len(train_data)*0.9)
                    train_data, val_data = train_data[:pivot], train_data[pivot:]
            else:
                raise FileNotFoundError

            train_data = [self.convert(x) for x in train_data]
            val_data= [self.convert(x) for x in val_data]

            self.train_data = SequenceDataset(train_data, self.tokenizer, max_length=self.max_length)
            self.val_data = SequenceDataset(val_data, self.tokenizer, max_length=self.max_length, mode="val")
            print("train_length:", len(self.train_data))
            print("valid_length:", len(self.val_data))
            self.train_dl = DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.train_data.collate_fn)

        elif stage==pl.trainer.states.TrainerFn.PREDICTING:
            with open(path.join(self.data_dir, self.predict_name)) as fin:
                predict_data = json.load(fin)
                predict_data = [{"origin":x} for x in predict_data]
            for item in predict_data:
                item["input"] = self.generate_prompt(item["origin"])
                item["output"] = "need prediction"
            self.predict_data = SequenceDataset(predict_data, self.tokenizer, max_length=self.max_length, mode="test")
            print("precition_length:", len(self.predict_data))

    def save_prediction(self, output, output_path):
        results = []
        for origin, pred in output:
            item = dict(**origin)
            item[self.predict_output_key] = pred
            results.append(item)
        with open(output_path, "w") as fout:
            json.dump(results, fout, indent=4, ensure_ascii=False)