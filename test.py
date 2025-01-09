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
from utils import ParameterAdjuster
from config import get_config

def main():
    """
    Description of the function main.
    """
    # Implement function logic here
    parser = get_config()
    updated_parser = ParameterAdjuster.adjust_parameters_with_gui(parser, save_path="output_params")
    
    print("Done!")
    
if __name__ == "__main__":
    main()