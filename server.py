# coding=utf-8
import sys
import os
import argparse

from omegaconf import OmegaConf

# 添加src目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   
sys.path.append(os.path.dirname(BASE_DIR))              # 将src目录添加到环境

from lm_model import CausalLMModel
from datas import load_datamodule
import arg_utils, utils

utils.set_random_seed(20200819)
os.environ["TOKENIZERS_PARALLELISM"] = "True"

def parse_args():
    WORKING_DIR = "."

    # 设置参数
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8080, help="server port (default: 16)")
    parser.add_argument("--batch_size", type=int, default=16, help="input batch size for predict (default: 16)")

    parser.add_argument("--model_config_name", type=str, required=True, help="model config file name")
    parser.add_argument("--data_config_name", type=str, required=True, help="model config file name")
    parser.add_argument("--num_beams", type=int, default=1, help="number of beam search in eval step (default: 1)")

    parser.add_argument("--ckpt_save_path", type=str, default="{}/weights".format(WORKING_DIR), help="ckpt_save_path")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="checkpoint file name for resume")
    args, extra_args = parser.parse_known_args()

    print('--------config----------')
    model_config = OmegaConf.load("config/model/"+args.model_config_name+".yml")
    data_config = OmegaConf.load("config/data/"+args.data_config_name+".yml")
    args.__setattr__("model_config", arg_utils.merge_model_config(extra_args, model_config))
    args.__setattr__("data_config", arg_utils.merge_data_config(extra_args, data_config))
    utils.print_config(args)
    print('--------config----------')

    return args

def main(args):

    print("start predict model ...")

    predict_checkpoint=os.path.join(args.ckpt_save_path ,args.resume_ckpt)   # 加载已保存的模型继续训练
    model = CausalLMModel.load_from_checkpoint(predict_checkpoint, config=args)
    datamodule = load_datamodule(args.data_config, model.tokenizer, args.batch_size, args.batch_size)
    model.eval()

    from fastapi import FastAPI, Request
    import uvicorn
    app = FastAPI()

    @app.post("/")
    async def create_item(request: Request):
        request_data = await request.json()
        items = request_data.get("items", [])
        dataloader = datamodule.predict_dataloader(items)
        outputs = []
        for batch_id, batch in enumerate(dataloader):
            model.predict_step(batch, batch_id, step_outputs = outputs, datamodule = datamodule)
        results = []
        for origin, pred in outputs:
            item = dict(**origin)
            item[datamodule.predict_output_key] = pred
            results.append(item)
        return {
            "items": results
        }

    uvicorn.run(app, host='0.0.0.0', port=args.port, workers=1)
        

if __name__ == '__main__':
    args = parse_args()
    main(args)