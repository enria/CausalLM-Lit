import torch
import requests
import json

class VQAConfidenceRewardModule():
    def __init__(self, api_url, reward_batch_size=8) -> None:
        self.api_url = api_url
        self.reward_batch_size = reward_batch_size

    def vqa_service(self, vq, image):
        url = self.api_url
        headers = {'Content-Type': 'application/json'}

        try:
            score = []
            for i in range(0, len(image), self.reward_batch_size):
                payload = json.dumps({
                    "vq": vq[i:i+self.reward_batch_size],
                    "image": image[i:i+self.reward_batch_size]
                })
                response = requests.request("POST", url, headers=headers, data=payload)
                response = response.json()
                batch_score = response["score"]
                score.extend(batch_score)
            return score
        except BaseException as e:
            print(e)
            return [0]*len(image)

    def __call__(self, output):
        vq, image = [],[]
        for prompt, pred, origin in output:
            pred = pred.strip()
            vq.append(pred)
            image.append(origin["image_path"])
        score = self.vqa_service(vq, image)
        rewards = [torch.tensor((s-0.5)*2) for s in score]
        return rewards
    

class VisualHintHelpfulRewardModule():
    prompt_template = ("Image Caption: {caption}\n"
    "{example}\n"
    "Question: {question}\n"
    "(Must return an answer. The final answer should be 1 or 2 words (maximum 2 words). If you are not sure, you can guess the most plausible answer)\n"
    "Answer: ")

    hits_template = "Visual Hints: 1. What the man is holding? a frisbee"

    system_setting = """Imagine you are a blind but intelligent question answering system. You are asked a visual question about an image. I will provide you the caption of the image and some useful visual hints. Please use your best judgement to answer the visual question. 
Examples: 
Image Caption: A man holding a dog on his back.
Visual Hints: 1. What the man is holding? a frisbee
Question: Which part of this animal would be in use of it was playing the game that is played with the items the man is holding? 
(Must return an answer. The final answer should be 1 or 2 words (maximum 2 words). If you are not sure, you can guess the most plausible answer) 
Answer: mouth
Image Caption: A busy city street with many people walking around.
Question: Why might someone go to this place?
(Must return an answer. The final answer should be 1 or 2 words (maximum 2 words). If you are not sure, you can guess the most plausible answer) 
Answer: shop"""

    def __init__(self, api_url, vqa_batch_size=8, llm_batch_size=4, llm="chatglm") -> None:
        self.api_url = api_url
        self.vqa_batch_size = vqa_batch_size
        self.llm_batch_size = llm_batch_size

        import sys
        sys.path.append("/home/yadong/workspace/okvqa/llm")
        sys.path.append("/home/yadong/workspace/okvqa/vqa4ok")

        from caption_vqa_llm import ChatGLMVQALLM, LLaMAVQALLM, ChatGPTVQALLM
        llm_cls = {
            ChatGLMVQALLM.name: ChatGLMVQALLM,
            LLaMAVQALLM.name: LLaMAVQALLM,
            ChatGPTVQALLM.name: ChatGPTVQALLM
        }

        # llm = ChatGLMVQALLM(prompt_template, system_setting)
        self.llm = llm_cls[llm](self.prompt_template, self.system_setting)

        sys.path.append("/home/yadong/workspace/okvqa/vqa4ok")
        from evaluation import sample_evaluate
        self.evaluate = sample_evaluate
    
    def is_no_question(self, question):
        return question.lower()=="the information in the caption is enough to answer the question"

    def vqa_service(self, items):
        url = self.api_url
        headers = {'Content-Type': 'application/json'}

        for i in range(0, len(items), self.vqa_batch_size):
            batch_items = items[i:i+self.vqa_batch_size]
            payload = json.dumps({
                "vq": [x["vq"] for x in batch_items],
                "image": [x["image_path"] for x in batch_items]
            })
            try:
                response = requests.request("POST", url, headers=headers, data=payload)
                response = response.json()
                for item, vqa, vqa_confidence in zip(batch_items, response["response"], response["score"]):
                    item["vqa"] = vqa
                    item["vqa_confidence"] = vqa_confidence
            except BaseException as e:
                print(e)
    
    def llm_vqa(self, items, use_hint=False):
        answers = []
        if self.llm_batch_size<=1:
            for sample in items:
                question = sample["question"]
                caption = sample["caption"]
                if use_hint: 
                    hints = [(i+1, q, a) for i,(q, a) in enumerate(zip([sample["vq"]], [sample["vqa"]]))]
                    hints_str = "Visual Hints: "+"\n".join([f"{i}. {q} {a}" for i,q,a in hints])
                else:
                    hints_str = ""
                try:
                    response = self.llm.inference(caption, question, hints_str, refine=False)
                except BaseException as e:
                    print(e)
                    response = ""
                answers.append(response)
        else:
            for i in range(0, len(items), self.llm_batch_size):
                batch = []
                for j in range(i, min(i+ self.llm_batch_size, len(items))):
                    sample = items[j]
                    question = sample["question"]
                    caption = sample["caption"]
                    if use_hint: 
                        hints = [(i+1, q, a) for i,(q, a) in enumerate(zip([sample["vq"]], [sample["vqa"]]))]
                        hints_str = "Visual Hints: "+"\n".join([f"{i}. {q} {a}" for i,q,a in hints])
                    else:
                        hints_str = ""
                    batch.append((caption, question, hints_str))
                try:
                    response = self.llm.batch_inference(batch, refine=False)
                except BaseException as e:
                    print(e)
                    response = [""]*len(batch)
                answers.extend(response)
        return answers
    
    def calculate_score(self, item, answer_key):
        return self.evaluate(item["answers"],item[answer_key])[1]
    

    def __call__(self, output):
        items = []
        item_index = {}
        for i, (prompt, pred, origin) in enumerate(output):
            pred = pred.strip()
            item = dict(**origin, **{"vq": pred})
            items.append(item)
            item_index[item["question_id"]] = i

        no_hint_items, need_hint_items = [], []
        for item in items:
            if self.is_no_question(item["vq"]):
                no_hint_items.append(item)
            else:
                need_hint_items.append(item)

        self.llm_vqa(items)
        self.vqa_service(need_hint_items)
        self.llm_vqa(need_hint_items, use_hint=True)

        for item in no_hint_items:
            vqa_score = self.calculate_score(item, "origin_answer")
            item["denoise_reward"] = (vqa_score-0.5)*2
            item["helpful_reward"] = 0
        
        for item in need_hint_items:
            origin_score = self.calculate_score(item, "origin_answer")
            hinted_score = self.calculate_score(item, "hinted_answer")
            item["helpful_reward"] = hinted_score-origin_score
            item["denoise_reward"] = 0
        
        rewards = [0]*len(items)
        for item in items:
            item["faithful_reward"] = (item["vqa_confidence"]-0.5)*2
            item["reward"] = item["helpful_reward"] + item["denoise_reward"]
            rewards[item_index[item["question_id"]]] = torch.tensor(item["reward"])

        return rewards