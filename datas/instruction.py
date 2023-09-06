import json
from os import path
from random import Random

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from .seq_data import SequenceDataset, SequenceDM
from .mixins import AccuracyDataMixin

class InstructionDM(SequenceDM):
    def __init__(self, **args):
        super().__init__(**args)

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
    
    def convert(self, item, default_output=None):
        return {
            "input":  self.generate_prompt(item),
            "output": item["output"] if default_output is None else item.get("output", default_output)
        }

class AccuracyDM(InstructionDM, AccuracyDataMixin):
    pass