
import torch
import numpy as np
def extract_weights(model):
    weights = []
    state_dict = model.to(torch.device('cpu')).state_dict()
    for name in state_dict.keys():
        weight = state_dict[name]
        weights.append((name, weight))
    return weights

def load_weights(model, weights):
    for i, p in enumerate(model.parameters()):
        weight=weights[0:p.data.numel()]
        p.data = weight.view(*p.shape)
        weights=weights[p.data.numel():]

def flatten_weights(weights):
    # Flatten weights into vectors
    weight_vecs = []
    for _, weight in weights:
        weight_vecs.extend(weight.flatten().tolist())
    return torch.tensor(weight_vecs)

def extract_grads(model):
    grads = []
    for name, weight in model.to(torch.device('cpu')).named_parameters():  # pylint: disable=no-member
        if weight.requires_grad:
            grads.append((name, weight.grad))
    return grads

def pack(model,info):
    weights_vec=flatten_weights(extract_weights(model))
    info_len=len(info)
    weights_vec=torch.cat((weights_vec,info),dim=0)
    return weights_vec, info_len

def unpack(weights_vec,info_len):
    weights=weights_vec[:-info_len]
    info=weights_vec[-info_len:]
    return weights, info

def extra_info(*args):
    return torch.tensor([arg for arg in args])

import random
import os
def setup_seed(seed):
    r"""
    Fix all the random seed used in numpy, torch and random module

    Args:
        seed (int): the random seed
    """
    if seed < 0:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        seed = -seed
    random.seed(1 + seed)
    np.random.seed(21 + seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(12 + seed)
    torch.cuda.manual_seed_all(123 + seed)