# coding=utf-8
import sys
import os
import argparse

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint,LearningRateMonitor
from lightning.pytorch.loggers.wandb import WandbLogger

from omegaconf import OmegaConf

# 添加src目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   
sys.path.append(os.path.dirname(BASE_DIR))              # 将src目录添加到环境

from lm_model import CausalLMModel
from datas import load_datamodule
from misc import utils

utils.set_random_seed(20200819)
os.environ["TOKENIZERS_PARALLELISM"] = "True"


def parse_args():
    WORKING_DIR = "."

    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="train", choices=["train","test","predict","data", "check"], help="the stage of learning process")
    parser.add_argument("--batch_size", type=int, default=16, help="input batch size for training (default: 16)")
    parser.add_argument("--val_batch_size", type=int, default=-1, help="input batch size for val and test (default: batch_size)")
    parser.add_argument("--max_epochs", type=int, default=20, help="the max epochs for training and test (default: 5)")
    parser.add_argument("--sanity_step", type=int, default=2, help="sanity check step (default: 2)")

    parser.add_argument("--model_config_name", type=str, required=True, help="model config file name")
    parser.add_argument("--data_config_name", type=str, required=True, help="model config file name")
    parser.add_argument("--accumulate_grads", type=int, default=4, help="accumulate Grads for train steps (default: 4)")
    parser.add_argument("--warmup_rate", type=float, default=0, help="warmup rate (default: 0)")
    parser.add_argument("--disable_linear_schedule", action='store_true', help="not use liner schedule")
    parser.add_argument("--num_beams", type=int, default=1, help="number of beam search in eval step (default: 1)")

    parser.add_argument("--train_num", type=int, default=-1,help="train data number")
    parser.add_argument("--dev_num", type=int, default=-1,help="train data number")

    parser.add_argument("--ckpt_name_prefix",  type=str, default="base", help="ckpt save name")
    parser.add_argument("--ckpt_save_path", type=str, default="{}/weights".format(WORKING_DIR), help="ckpt_save_path")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="checkpoint file name for resume")
    parser.add_argument("--rename_best_ckpt", action='store_true', help="rename best ckpt (default to best.ckpt)")
    parser.add_argument("--save_full_model", action='store_true', help="store full model")

    parser.add_argument("--predict_output_dir",  type=str, default="output", help="prediction output name")
    parser.add_argument("--predict_output_name",  type=str, default=None, help="prediction output name")

    parser.add_argument("--project",  type=str, default=None, help="wandb project name")
    parser.add_argument("--run_name",  type=str, default="base", help="wandb run name")
    parser.add_argument("--log_every_n_steps",  type=int, default=10, help="lightning log_every_n_steps")

    args, extra_args = parser.parse_known_args()

    print('--------config----------')
    model_config = OmegaConf.load("config/model/"+args.model_config_name+".yml")
    data_config = OmegaConf.load("config/data/"+args.data_config_name+".yml")
    args.__setattr__("model_config", utils.merge_model_config(extra_args, model_config))
    args.__setattr__("data_config", utils.merge_data_config(extra_args, data_config))
    utils.print_config(args)
    print('--------config----------')

    return args

def main(args):
    model = CausalLMModel(args)
    data_module = load_datamodule(args.data_config, model.tokenizer, args.batch_size, args.val_batch_size, args.train_num, args.dev_num)

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
                mode = args.data_config.metrics.get("mode", "max"),
                save_top_k = 3,                                          # 保存得分最高的前两个模型
                verbose = True
            )
            early_stopping=EarlyStopping(metric, mode="max",patience=3)
        else:
            ckpt_callback = ModelCheckpoint(
                dirpath = args.ckpt_save_path,                           # 模型保存路径
                filename = args.ckpt_name_prefix + "_{val_loss:.3f}_{epoch}",  # 模型保存名称，参数ckpt_name后加入epoch信息以及验证集分数
                monitor = "val_loss",                                        # 根据验证集上的准确率评估模型优劣
                mode = 'min',
                save_top_k = 3,                                          # 保存得分最高的前两个模型
                verbose = True
            )
            early_stopping=EarlyStopping("val_loss", mode="min",patience=4)
        
        lr_logger = LearningRateMonitor()
        callbacks = [ckpt_callback, early_stopping, lr_logger]
        
        if args.rename_best_ckpt:
            from utils.callback import RenameBestCheckpointCallback
            best_ckpt_callback = RenameBestCheckpointCallback(ckpt_callback)
            callbacks.append(best_ckpt_callback)
        
        resume_checkpoint=None
        if args.resume_ckpt:
            resume_checkpoint=os.path.join(args.ckpt_save_path ,args.resume_ckpt)   # 加载已保存的模型继续训练

        run_name = args.run_name
        if run_name == "base":
            run_name = f"{args.data_config_name}({args.model_config_name})"
        logger = WandbLogger(name=run_name, project=args.project, offline=args.project==None, config=args)

        # 设置训练器
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            callbacks=callbacks,
            logger = logger,
            log_every_n_steps=args.log_every_n_steps,
            accumulate_grad_batches=args.accumulate_grads,
            num_sanity_val_steps=args.sanity_step
        )

        # 开始训练模型
        trainer.fit(model, ckpt_path=resume_checkpoint, datamodule=data_module)

        if args.save_full_model:
            model.model.save_pretrained(ckpt_callback.best_model_path+".full")

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

        predict_checkpoint=os.path.join(args.ckpt_save_path ,args.resume_ckpt)   # 加载已保存的模型继续训练
        # predict_checkpoint = None

        # 设置训练器
        trainer = pl.Trainer(devices=1)

        # 开始预测结果
        trainer.predict(model, ckpt_path=predict_checkpoint, datamodule=data_module)
    
    elif args.stage=="data": # 检查数据分词后的长度，方便去设置 datamoduel max_length
        from tqdm import tqdm
        data_module.setup("fit")
        length = []
        for batch in tqdm(data_module.train_dataloader()):
            length.append(batch["input_ids"].shape[1])

        import matplotlib.pyplot as plt
        plt.hist(length, bins=100)
        plt.title('Histogram')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.savefig("output/input_length.svg")
        

if __name__ == '__main__':
    args = parse_args()
    main(args)