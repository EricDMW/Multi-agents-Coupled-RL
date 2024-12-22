#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File   : set_seed.py
@Author : Dongming Wang
@Email  : dongming.wang@email.ucr.edu
@Project: $(basename ~/project/coupled_rl/utils)
@Date   : 12/21/2024
@Time   : 17:24:24
@Info   : Description of the script
"""
import numpy as np
import torch
import random

def set_seed(seed):
 
    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)

    # Set the seed for GPU (if using CUDA)
    torch.cuda.manual_seed_all(seed)

    # Set the seed for Python's random module
    random.seed(seed)

    # For reproducibility with CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

