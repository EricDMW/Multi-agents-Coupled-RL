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

# Create an empty tensor (shape: 0)
tensor = torch.empty(0)

# Create a 2x3 tensor
tensor_new = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Convert scalar 'a' to a 1D tensor and reshape it to match the second dimension of tensor_new
tensor_a = torch.tensor([7, 8, 8]).unsqueeze(0)  # Unsqueeze to make it shape (1, 3)

# Now you can concatenate along axis 0 (rows)
result = torch.cat([tensor_new, tensor_a], dim=0)

print(result)

