#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File   : train.py
@Author : Dongming Wang
@Email  : dongming.wang@email.ucr.edu
@Project: $(basename ~/project/coupled_rl/algorithm)
@Date   : 12/12/2024
@Time   : 20:11:10
@Info   : Description of the script
"""

import sys
import torch
import os
import setproctitle
import tqdm

from pathlib import Path
from tqdm import tqdm, trange

# avoid potential import error
parents_path = Path(__file__).parents[1]
sys.path.append(str(parents_path))

# parameters 
from config import get_config
from utils.device_setting import get_device

from asymmetric import Asymmetric
from env import SyntheticEnv

def train():
    """
    Description of the function train.
    """
    ### Training preparation: parameters and device setting
    
    # set the running device
    device = get_device()
    
    # parser parameters
    parser = get_config()
    para = parser.parse_args()
    
    # buid the environment
    
    
    # initialize the policy
    policy = Asymmetric(para,device)
    
    # initialize the environment
    env = SyntheticEnv(para,device)
    
    
    # training progress
    for _ in trange(para.episode_num):
        
        env.reset()
        
        for _ in range(para.steps_num):
            actions = torch.zeros(size=(para.agents_num, ), device=device, dtype=torch.bool)
            env.step(actions)
    
    policy.save_model()
    
    
    

if __name__ == "__main__":
    train()