# coding=utf-8
import sys
import os
import argparse

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint,LearningRateMonitor
from lightning.pytorch.loggers.wandb import WandbLogger

from omegaconf import OmegaConf

from callbacks import RenameBestCheckpointCallback

# 添加src目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   
sys.path.append(os.path.dirname(BASE_DIR))              # 将src目录添加到环境

from rl_model import ReinforcementLMModel
from datas import load_datamodule
import arg_utils, utils

utils.set_random_seed(20200819)
os.environ["TOKENIZERS_PARALLELISM"] = "True"


def parse_args():
    WORKING_DIR = "."

    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="train", choices=["train","test","predict","check"], help="the stage of learning process")
    parser.add_argument("--batch_size", type=int, default=16, help="input batch size for training  (default: 8)")
    parser.add_argument("--val_batch_size", type=int, default=-1, help="input batch size for validation and test (default: 8)")
    parser.add_argument("--mini_batch_size", type=int, default=16, help="batch size for ppo optimization (default: 8)")
    parser.add_argument("--max_epochs", type=int, default=20, help="the max epochs for training and test (default: 5)")

    parser.add_argument("--model_config_name", type=str, required=True, help="model config file name")
    parser.add_argument("--data_config_name", type=str, required=True, help="model config file name")
    parser.add_argument("--accumulate_grads", type=int, default=4, help="accumulate Grads for train steps (default: 4)")
    parser.add_argument("--warmup_rate", type=float, default=0.1, help="warmup rate (default: 0.1)")
    parser.add_argument("--num_beams", type=int, default=1, help="number of beam search in eval step (default: 1)")

    parser.add_argument("--train_num", type=int, default=-1,help="train data number")
    parser.add_argument("--dev_num", type=int, default=-1,help="train data number")

    parser.add_argument("--ckpt_name_prefix",  type=str, default="base", help="ckpt save name")
    parser.add_argument("--ckpt_save_path", type=str, default="{}/weights".format(WORKING_DIR), help="ckpt_save_path")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="checkpoint file name for resume")

    parser.add_argument("--predict_output_dir",  type=str, default="output", help="prediction output name")
    parser.add_argument("--predict_output_name",  type=str, default=None, help="prediction output name")

    parser.add_argument("--project",  type=str, default=None, help="wandb project name")
    parser.add_argument("--run_name",  type=str, default="base", help="wandb run name")

    args, extra_args = parser.parse_known_args()

    print('--------config----------')
    model_config = OmegaConf.load("config/model/"+args.model_config_name+".yml")
    data_config = OmegaConf.load("config/data/"+args.data_config_name+".yml")
    args.__setattr__("model_config", arg_utils.merge_model_config(extra_args, model_config))
    args.__setattr__("data_config", arg_utils.merge_data_config(extra_args, data_config))

    print(args)
    print('--------config----------')

    return args

def main(args):
    model = ReinforcementLMModel(args)
    model.automatic_optimization = False
    data_module = load_datamodule(args.data_config, model.tokenizer, args.batch_size, args.val_batch_size)

    if args.stage == "train":
        # ============= train 训练模型==============
        print("start train model ...")
        # 设置保存模型的路径及参数

        if model.calculate_metric:
            metric = args.data_config.metrics.name
            ckpt_callback = ModelCheckpoint(
                dirpath = args.ckpt_save_path,                           # 模型保存路径
                filename = args.ckpt_name_prefix + "_{%s:.3f}_{epoch}"%metric,  # 模型保存名称，参数ckpt_name后加入epoch信息以及验证集分数
                monitor = metric,                                        # 根据验证集上的准确率评估模型优劣
                mode = 'max',
                save_top_k = 3,                                          # 保存得分最高的前两个模型
                verbose = True
            )
            early_stopping=EarlyStopping(metric, mode="max",patience=4)
        else:
            ckpt_callback = ModelCheckpoint(
                dirpath = args.ckpt_save_path,                           # 模型保存路径
                filename = args.ckpt_name_prefix + "_{val_reward:.3f}_{epoch}",  # 模型保存名称，参数ckpt_name后加入epoch信息以及验证集分数
                monitor = "val_reward",                                        # 根据验证集上的准确率评估模型优劣
                mode = 'max',
                save_top_k = 3,                                          # 保存得分最高的前两个模型
                verbose = True
            )
            early_stopping=EarlyStopping("val_reward", mode="max",patience=4)
        
        resume_checkpoint=None
        if args.resume_ckpt:
            resume_checkpoint=os.path.join(args.ckpt_save_path ,args.resume_ckpt)   # 加载已保存的模型继续训练
        
        # best_ckpt_callback = RenameBestCheckpointCallback(ckpt_callback)

        run_name = args.run_name
        if run_name == "base":
            run_name = f"{args.data_config_name}({args.model_config_name})"
        logger = WandbLogger(name=run_name, project=args.project, offline=args.project==None)

        # 设置训练器
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            logger = logger,
            callbacks=[ckpt_callback, early_stopping],
            devices=1,
            log_every_n_steps=1
        )

        # 开始训练模型
        trainer.fit(model, ckpt_path=resume_checkpoint, datamodule=data_module)

        ckpt_callback.best_model_path

    elif args.stage=="test":
        print("start test model ...")

        resume_checkpoint=None
        if args.resume_ckpt:
            resume_checkpoint=os.path.join(args.ckpt_save_path ,args.resume_ckpt)   # 加载已保存的模型继续训练

        # 设置训练器
        trainer = pl.Trainer( devices=1 , logger=False)

        # 开始预测结果
        trainer.test(model,ckpt_path=resume_checkpoint, datamodule=data_module)

    elif args.stage=="predict":
        print("start test model ...")
        if not args.predict_output_name:
            args.predict_output_name = args.resume_ckpt.rsplit(".",1)[0]+".json"

        resume_checkpoint=os.path.join(args.ckpt_save_path ,args.resume_ckpt)   # 加载已保存的模型继续训练

        # 设置训练器
        trainer = pl.Trainer(devices=1)

        # 开始预测结果
        trainer.predict(model,ckpt_path=resume_checkpoint, datamodule=data_module)
    
    elif args.stage=="check":
        pass

if __name__ == '__main__':
    args = parse_args()
    main(args)