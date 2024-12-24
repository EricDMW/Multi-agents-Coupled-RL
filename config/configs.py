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
        "--is_training",
        type=bool,
        default=True,
        help="determine use stochatic policy (training) or greedy policy (execution)"
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

    parser.add_argument(
        "--state_space_size",
        type=int,
        default=2,
        help="size of the state space of each agents"
    )
    
    parser.add_argument(
        "--action_space_size",
        type=int,
        default=2,
        help="action space size of each agents"
    )
    
    # algorithm setting parameters
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.8,
        help="discount factor"
    )
    
    parser.add_argument(
        "--kappa_o",
        type = int,
        default=1,
        help="Order of neighbors used"
    )
    
    parser.add_argument(
        "--kappa_r",
        type = int,
        default=1,
        help="Order of neighbors used"
    )
    
    
    
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=1,
        help="the random seed for torch"
    )
    
    parser.add_argument(
        "--fix_seed",
        type=bool,
        default=True,
        help="use fixed random seed or not"
    )
    
    
    # frequently used parameters
    parser.add_argument(
        "--h",
        type=float,
        default=2,
        # performance tested: the leftmost is current testing one, the rest part ordered by performance
        choices=[2,1,3],
        help="step size prameter h"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=2,
        # performance tested: the leftmost is current testing one, the rest part ordered by performance
        choices=[2,1,3],
        help="step size prameter eta"
    )
    parser.add_argument(
        "--t_0",
        type=int,
        default=5,
        help="step size prameter t_0"
    )
    
    parser.add_argument(
        "--steps_num",
        type=int,
        default=int(1e3),
        help="lenth of runing steps of each episode"
    )
    
    parser.add_argument(
        "--episode_num",
        type=int,
        default=int(1e3),
        help="number of training episodes",
    )
    parser.add_argument(
        "--save_frequency",
        type=int,
        default=100,
        help="save the model during training process"
    )   
    
    return parser
