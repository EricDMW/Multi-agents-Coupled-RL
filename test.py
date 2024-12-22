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

# Assume self.policy_para_tensor is a 2D tensor
policy_para_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Take a view from self.policy_para_tensor
theta_to_update = policy_para_tensor[0, 1]  # This will be 2

# Modify the view
theta_to_update = 10

# Now self.policy_para_tensor will also reflect this change
print(policy_para_tensor)

