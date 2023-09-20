import json
import re
from copy import deepcopy
from os import path
from random import Random

import lightning.pytorch as pl

from .seq_data import SequenceDM, PredictionSaveMixin
from .mixins import MetricDataMixin




class Metric:
    """ Tuple Metric
        author: UIE
    """
    def __init__(self, match_mode='normal'):
        self.tp = 0.
        self.gold_num = 0.
        self.pred_num = 0.
        self.match_mode = match_mode
        assert self.match_mode in {'set', 'normal', 'multimatch'}

    @staticmethod
    def safe_div(a, b):
        return 0. if b == 0. else a / b

    def compute_f1(self, prefix=''):
        tp = self.tp
        pred_num = self.pred_num
        gold_num = self.gold_num
        p, r = self.safe_div(tp, pred_num), self.safe_div(tp, gold_num)
        return {prefix + 'tp': tp,
                prefix + 'gold': gold_num,
                prefix + 'pred': pred_num,
                prefix + 'precision': p * 100,
                prefix + 'recall': r * 100,
                prefix + 'f1': self.safe_div(2 * p * r, p + r) * 100
                }

    def count_instance(self, gold_list, pred_list):
        if self.match_mode == 'set':
            gold_list = set(gold_list)
            pred_list = set(pred_list)
            self.gold_num += len(gold_list)
            self.pred_num += len(pred_list)
            self.tp += len(gold_list & pred_list)

        else:
            self.gold_num += len(gold_list)
            self.pred_num += len(pred_list)

            if len(gold_list) > 0 and len(pred_list) > 0:
                # guarantee length same
                assert len(gold_list[0]) == len(pred_list[0])

            dup_gold_list = deepcopy(gold_list)
            for pred in pred_list:
                if pred in dup_gold_list:
                    self.tp += 1
                    if self.match_mode == 'normal':
                        # Each Gold Instance can be matched one time
                        dup_gold_list.remove(pred)

class DuEEFinDM(SequenceDM, MetricDataMixin):

    def setup(self, stage: str):
        if stage==pl.trainer.states.TrainerFn.TESTING:
            super().setup(pl.trainer.states.TrainerFn.FITTING)
            self.test_data = self.val_data
        else:
            super().setup(stage)

    def record_to_seq(self, records):
        seq = []
        for record in records:
            record_seq = f"{record['event_type']}： "
            arg_seq = []
            for arg in record["arguments"]:
                arg_seq.append(f"{arg['argument']}[{arg['role']}]")
            record_seq+="，".join(arg_seq)
            seq.append(record_seq)
        return "###".join(seq)
    
    def seq_to_record(self, seq):
        records = []
        for record in seq.split("###"):
            if "：" not in record: continue
            event_type, arg_seq= record.split("：",1)
            args = []
            for arg in arg_seq.split("，"):
                arg = arg.strip()
                match = re.match("(.+)\[(.+)\]",arg)
                if match: args.append({
                    "argument": match.groups()[0],
                    "role": match.groups()[1]
                })
                    
            records.append({
                "event_type": event_type,
                "arguments": args
            })
        return records
    
    def seq_to_args(self, seq):
        records = self.seq_to_record(seq)
        args = []
        for record in records:
            for arg in record["arguments"]:
                args.append((record["event_type"], arg["role"], arg["argument"]))
        return args
    
    def calculate_instance(self, gold_seq, pred_seq, metric):
        gold_args = self.seq_to_args(gold_seq)
        pred_args = self.seq_to_args(pred_seq)
        metric.count_instance(gold_args, pred_args)
    
    def calculate_metrics(self, output):
        metric = Metric()
        for gold, pred, origin in output:
            self.calculate_instance(gold, pred, metric)
        return metric.compute_f1()
    
    def convert(self, item):
        if "event_list" not in item:
            return None
        sample =  {
            "input":  f"标题：{item['title']}\n 新闻内容: {item['text']}\n 事件列表:\n",
            "output": self.record_to_seq(item["event_list"]),
            "origin": item
        }
        return sample
    
    def load_data(self, file):
        data = []
        with open(file) as fin:
            for line in fin:
                data.append(json.loads(line))
        return data
    