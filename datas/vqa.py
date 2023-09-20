import os
import os.path as path
import json
from random import Random

import numpy as np
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from .seq_data import SequenceDataset, SequenceDM
from .mixins import ClassificationDataMixin, MetricDataMixin

class VQAMergeDM(SequenceDM, ClassificationDataMixin):
    def __init__(self, **args):
        super().__init__(**args)
        self.set_category(["visual", "reasoning", "both", "none"])
    
    def generate_prompt(self,context):
        template = ("The following is an instruction that describes the task of question tendency identification "
                    "### Instruction: Is this a reasoning question or a visual question? "
                    "### Input: {context} "
                    "### Response: ")
        return template.format(context=context)
    
    def convert(self, item):
        if item["output"]=="none": return None # filter this type data
        return {
            "input":  self.generate_prompt(context=item["input"]),
            "output": item["output"]
        }
    
class OKVQADM(SequenceDM, MetricDataMixin):
    def __init__(self, caption_type="general", **args):
        super().__init__(**args)
        self.caption_type = caption_type

    def generate_prompt(self, item):
        template = ("Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request. "
                    "### Instruction: {instruction} "
                    "### Input: {input} "
                    "### Response: "
        )
        return template.format(instruction=item["question"], input=item["caption"])
    
    def convert(self, item, keep_origin=False):
        sample = { 
            "input":  self.generate_prompt(item),
            "output": item["answer"] 
        }
        if keep_origin:
            sample["origin"] = item
        return sample
    def load_samples(self, keep_origin=False):
        question_file = path.join(self.data_dir, "questions.json")
        caption_file = path.join(self.data_dir, f"{self.caption_type}_caption.json")
        answer_file = path.join(self.data_dir, "annotations.json")

        with open(question_file) as fin:
            samples = json.load(fin)["questions"]
        with open(caption_file) as fin:
            samples_caption = json.load(fin)

        with open(answer_file) as fin:
            answers = json.load(fin)["annotations"]
            from collections import Counter
            def most_answer(answers):
                counter = Counter(answers)
                return counter.most_common(1)[0][0]
            common_answer_index = {x["question_id"]:most_answer([y["answer"] for y in x["answers"]]) for x in answers}
            all_answer_index = {x["question_id"]:[y["answer"] for y in x["answers"]] for x in answers}

        for item in samples:
            item["caption"] = samples_caption[str(item['question_id'])]["caption"]
            item["answer"] = common_answer_index[item['question_id']]
            item["answers"] = all_answer_index[item['question_id']]

        samples = [self.convert(x, keep_origin) for x in samples]

        return samples

    def setup(self, stage: str):
        if stage in [pl.trainer.states.TrainerFn.FITTING, pl.trainer.states.TrainerFn.TESTING]:
            train_data = self.load_samples(keep_origin=True)

            shuffle = Random(42).shuffle
            shuffle(train_data)
            pivot = int(len(train_data)*0.9)
            train_data, val_data = train_data[:pivot], train_data[pivot:]

            self.train_data = SequenceDataset(train_data, self.tokenizer, max_length=442, input_truncation_side="right")
            self.val_data = SequenceDataset(val_data, self.tokenizer, max_length=442, mode="eval", input_truncation_side="right")
            print("train_length:", len(self.train_data))
            print("valid_length:", len(self.val_data))
            self.train_dl = DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.train_data.collate_fn)
            if stage == pl.trainer.states.TrainerFn.TESTING:
                self.test_data = self.val_data

        elif stage==pl.trainer.states.TrainerFn.PREDICTING:
            predict_data = self.load_samples(keep_origin=True)
            self.predict_data = SequenceDataset(predict_data, self.tokenizer, max_length=442, mode="predict")
            print("precition_length:", len(self.predict_data))

    def calculate_metrics(self, output):
        accs = []
        for gold, pred, origin in output:
            pred = pred.strip()
            answers= origin["answers"]
            counter = 0
            for ii in range(len(answers)):
                if pred == answers[ii]: counter+=1
            accs.append(min(1.,float(counter)*0.3))
        return {"vqa_acc": np.mean(accs)*100}
    