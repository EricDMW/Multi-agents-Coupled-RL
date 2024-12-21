#!opt/anaconda3/envs/lie_optimal/bin python
# _*_ coding: utf-8 _*_
'''
@File  : device_setting.py
@Author: Dongming Wang
@Email : dongming.wang@email.ucr.edu
@Date  : 12/21/2024
@Time  : 13:02:52
@Info  : TBC
'''

import torch


def  get_device():
    """
    Function:  function_name
    Description: TBC
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    return device
