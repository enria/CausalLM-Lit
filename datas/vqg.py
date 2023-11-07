import json

from tqdm import tqdm

from .rl_data import RLDM
from .seq_data import SequenceDM, PredictionSaveMixin


class CaptionVQGDM(SequenceDM):
    def convert(self, item, keep_origin=True):
        sample =  {
            "input":  f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n",
            "output": f"{item['output']}",
            "origin": item
        }
        return sample

class ConfidentVQGDM(RLDM):
    def convert(self, item):
        sample =  {
            "input":  f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n",
            "origin": item
        }
        return sample

class HelpfulVQGDM(RLDM, PredictionSaveMixin):

    def convert(self, item):
        sample =  {
            "input":  f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n",
            "origin": item
        }
        return sample
    
    def calculate_metrics(self, output, reward_model):
        rewards, envs = reward_model(output)
        return {
            "vqa_acc": envs["env/vqa_acc"]*100,
            "origin_vqa_acc": envs["env/origin_vqa_acc"]*100
        }
    
    def save_prediction(self, output, output_path, reward_model):
        items, batch = [], []
        batch_size = 256
        for i, (origin, pred) in tqdm(enumerate(output), total=len(output)):
            pred = pred.strip()
            item = dict(**origin, **{"vq": pred})
            items.append(item)
            batch.append(item)

            if len(batch)==batch_size or i==len(output)-1:
                reward_model.vqa_service(batch)
                # reward_model.llm_vqa(batch)
                # reward_model.llm_vqa(batch, use_hint=True)
                batch.clear()

        with open(output_path, "w") as fout:
            json.dump(items, fout, indent=4, ensure_ascii=False)