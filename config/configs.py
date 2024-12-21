#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File   : configs.py
@Author : Dongming Wang
@Email  : dongming.wang@email.ucr.edu
@Project: $(basename ~/project/coupled_rl/config)
@Date   : 12/12/2024
@Time   : 20:13:27
@Info   : parameters used
"""

import argparse


def get_config():
    """
    Parameters Used:
    """

    parser = argparse.ArgumentParser(
        description="Coupled_RL", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    
    # general parameters
    parser.add_argument(
        "--env_name",
        type=str,
        default="Synthetic",
        help="Name of the environment."
    )
    
    # operator parameters
    parser.add_argument(
         "--agents_num",
        type=int,
        default=4,
        help="Number of agents.",
    )
    
    return parser
