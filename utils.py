import torch
import numpy as np
import random
import time

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def chain_get(data, keys, default):
    if len(keys)==0: return default
    if len(keys)==1: return data.get(keys[0], default)
    if keys[0] not in data: return default
    return chain_get(data[keys[0]], keys[1:], default)


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