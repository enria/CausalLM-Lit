import torch
from transformers import pipeline

class PositiveSentimentRewardModule():
    def __init__(self) -> None:
        self.sent_kwargs = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "batch_size": 16
        }

        self.pipeline = pipeline("sentiment-analysis","/pretrains/huggingface/lvwerra/distilbert-imdb")
    
    def __call__(self, output):
        texts = []
        for prompt, pred, _ in output:
            pred = pred.strip()
            texts.append(prompt + " " + pred)
        pipe_outputs = self.pipeline(texts, **self.sent_kwargs)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
        return rewards, {}