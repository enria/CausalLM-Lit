import os
import pathlib
import sys
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from transformers import AutoTokenizer

from bert_score.utils import (bert_cos_score_idf, cache_scibert, get_bert_embedding,
                    get_hash, get_idf_dict, get_model, get_tokenizer,
                    lang2model, model2layers, sent_encode)

from bert_score.utils import get_tokenizer, get_model
import torch
model_type = "/storage/pretrains/huggingface/bert-base-uncased"
tokenizer = get_tokenizer(model_type, False)
model = get_model(model_type, 9, False)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def bertscore(
    cands,
    refs,
    verbose=False,
    idf=False,
    batch_size=64,
    nthreads=4,
    all_layers=False
):
    assert len(cands) == len(refs), "Different number of candidates and references"

    if not idf:
        idf_dict = defaultdict(lambda: 1.0)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0
    elif isinstance(idf, dict):
        idf_dict = idf
    else:
        idf_dict = get_idf_dict(refs, tokenizer, nthreads=nthreads)

    all_preds = bert_cos_score_idf(
        model,
        refs,
        cands,
        tokenizer,
        idf_dict,
        verbose=verbose,
        device=device,
        batch_size=batch_size,
        all_layers=all_layers,
    ).cpu()


    out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F
    return out