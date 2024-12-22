#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File   : test.py
@Author : Dongming Wang
@Email  : dongming.wang@email.ucr.edu
@Project: $(basename ~/project/coupled_rl)
@Date   : 12/12/2024
@Time   : 20:21:33
@Info   : Description of the script
"""
import torch
import pandas as pd
import numpy as np

import torch


import torch

# Tensor a with indices (each row represents an index into b)
a = torch.tensor([1, 2, 3])

# Tensor b (5x5x5 tensor)
b = torch.arange(5*5*5).reshape(5, 5, 5)

# Use a to index into b and retrieve the values at the specified indices
result = b[a[ 0], a[1], a[2]]

print("Tensor b:\n", b)
print("Indices in a:\n", a)
print("Values at the indices:\n", result)
