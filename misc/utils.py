import torch
import numpy as np
import random
import time
import argparse
from omegaconf import OmegaConf, DictConfig

def print_config(config):
    from omegaconf import OmegaConf
    import yaml
    from copy import deepcopy
    config_dict = deepcopy(vars(config))
    if "data_config" in config_dict:
        config_dict["data_config"] = OmegaConf.to_container(config_dict["data_config"], resolve=True)
    if "model_config" in config_dict:
        config_dict["model_config"] = OmegaConf.to_container(config_dict["model_config"], resolve=True)
    print(yaml.dump(config_dict))

def chain_get(data, keys, default=None):
    if len(keys)==0: return default
    if len(keys)==1: return data.get(keys[0], default)
    if keys[0] not in data: return default
    return chain_get(data[keys[0]], keys[1:], default)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

class Timer:
    def __init__(self, record=None, key=None):
        self._start_time = None
        self.record = record
        self.key = key
        if record is not None:
            assert type(record)==dict
            assert key is not None

    def _start(self):
        """Start a new timer"""
        self._start_time = time.perf_counter()

    def _stop(self):
        """Stop the timer, and report the elapsed time"""
        self.elapsed_time = time.perf_counter() - self._start_time
        if self.record is not None:
            self.record[self.key] = self.elapsed_time

    def __enter__(self):
        """Start a new timer as a context manager"""
        self._start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self._stop()


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