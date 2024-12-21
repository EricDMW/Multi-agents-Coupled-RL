#!opt/anaconda3/envs/lie_optimal/bin python
# _*_ coding: utf-8 _*_
'''
@File  : synthetic_env.py
@Author: Dongming Wang
@Email : dongming.wang@email.ucr.edu
@Date  : 12/21/2024
@Time  : 13:07:02
@Info  : TBC
'''

import torch

import sys
from pathlib import Path

parents_path = Path(__file__).parents[1]
sys.path.append(str(parents_path))

from config import get_config
from utils.device_setting import get_device

class SyntheticEnv():
    def __init__(self, elements, operation):
        """
        Initialize the SyntheticEnv():.
        """