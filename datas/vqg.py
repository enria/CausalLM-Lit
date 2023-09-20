import json

from tqdm import tqdm

from .rl_data import RLDM
from .seq_data import SequenceDM, PredictionSaveMixin


class CaptionVQGDM(SequenceDM):
    def convert(self, item):
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
                vqa_anwsers = reward_model.vqa_service([i["vq"] for i in batch], [i["image_path"] for i in batch])
                for (vqa_answer, vqa_confidence), item in zip(vqa_anwsers, batch):
                    item["vqa"] = vqa_answer
                    item["vqa_confidence"] = vqa_confidence
                hinted_llm_answers = reward_model.llm_vqa(batch, use_hint=True)
                for answer, item in zip(hinted_llm_answers, batch):
                    item["answer"] = answer
                
                batch.clear()

        with open(output_path, "w") as fout:
            json.dump(items, fout, indent=4, ensure_ascii=False)