#!/usr/bin/env python
import argparse

from omegaconf import OmegaConf, DictConfig


def str2bool(v):
    '''
    将字符转化为bool类型
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def merge_model_config(args, model_config: DictConfig):
    conf = OmegaConf.from_cli(args)
    if "--M" in conf:
        model_config.merge_with(conf["--M"])
    return model_config

def merge_data_config(args, data_config: DictConfig):
    conf = OmegaConf.from_cli(args)
    if "--D" in conf:
        data_config.merge_with(conf["--D"])
    return data_config
