import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model, trainable_only=True):
    if trainable_only:
        parameters = (param for param in model.parameters() if param.requires_grad)
    else:
        parameters = model.parameters()

    return sum(param.numel() for param in parameters)


def resolve_device(prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
