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
    
    # training parameters
    parser.add_argument(
        "--episode_num",
        type=int,
        default=200,
        help="number of training episodes",
    )

    
    
    # parameters
    parser.add_argument(
        "--agents_num",
        type=int,
        default=12,
        help="Number of agents.",
    )
    
    parser.add_argument(
        "--trans_prob",
        type=float,
        default=0.8,
        help="The transition probability of agents 2->n-1 when s_i+1 = 0, a_i = 1"
    )

    
    # algorithm setting parameters
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor"
    )
    
    parser.add_argument(
        "--kappa",
        type = int,
        default=1,
        help="Order of neighbors used"
    )
    
    parser.add_argument(
        "--steps_num",
        type=int,
        default=50,
        help="lenth of runing steps of each episode"
    )
    
    return parser
