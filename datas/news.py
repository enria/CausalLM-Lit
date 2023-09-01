import os
import json
from random import Random

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from .seq_data import SequenceDataset, SequenceDM
from .mixins import ClassificationDataMixin

class NewsDM(SequenceDM, ClassificationDataMixin):
    def __init__(self, **args):
        super().__init__(**args)

        with open(os.path.join(self.data_dir, "News_Category.json")) as fin:
            category = json.load(fin)
            self.set_category(category)
    
    def generate_prompt(self,context):
        template = ("The following is an instruction that describes the task of text classification "
                    "### Instruction: Analyze the sources or types of the following news "
                    "### Input: {context} "
                    "### Response: "
        )
        return template.format(context=context)
    
    def convert(self, item):
        return {
            "input":  self.generate_prompt(context=item["headline"]+" "+item["short_description"]),
            "output": item["category"]
        }

    def setup(self, stage: str):
        data = []
        with open(os.path.join(self.data_dir, "News_Category_Dataset_v3.json")) as fin: 
            for line in fin:
                item = json.loads(line)
                data.append(self.convert(item))
        shuffle = Random(42).shuffle
        shuffle(data)
    
        if stage==pl.trainer.states.TrainerFn.FITTING:
            self.train_data = SequenceDataset(data[:9000], self.tokenizer, max_length=256)
            self.val_data = SequenceDataset(data[-1000:], self.tokenizer, max_length=256, mode="eval")
            self.train_dl = DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.train_data.collate_fn)
            print("train_length:", len(self.train_data))
            print("valid_length:", len(self.val_data))

        elif stage==pl.trainer.states.RunningStage.TESTING:
            self.test_data = SequenceDataset(data[-1000:], self.tokenizer, max_length=256, mode="test")
            print("test_length:", len(self.test_data))