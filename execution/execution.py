#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File   : execution.py
@Author : Dongming Wang
@Email  : dongming.wang@email.ucr.edu
@Project: $(basename ~/project/coupled_rl/execution)
@Date   : 12/21/2024
@Time   : 17:04:18
@Info   : Description of the script
"""
import sys
import torch
import re


from pathlib import Path
from tqdm import trange

# avoid potential import error
parents_path = Path(__file__).parents[1]
sys.path.append(str(parents_path))

# parameters 

from utils import set_seed


from algorithm import Asymmetric, initialize_para
from env import SyntheticEnv

class Execution:
    def __init__(self, *args, **kwargs):
        """
        Constructor for Execution
        """
        # Initialize class Execution attributes here
        # set the running device
        # wrong output of the 
        self.device, self.para = initialize_para()
        
        # modify execution parameters: execution mode, use greedy policy
        self.para.is_training = False
                
        ### buid the environment
        
        # initialize the policy
        self.policy = Asymmetric(self.para,self.device)
        
        # initialize the environment
        self.env = SyntheticEnv(self.para,self.device)
        
    def execution(self, file_path, running_step, *args):
        
        # set the seed for rep
        set_seed(self.para.rand_seed)
        
        # Load the tensor from the .pt file
        loaded_tensor = torch.load(file_path, map_location=self.device, weights_only=True)

        # Ensure the shapes match before updating
        if self.policy.policy_para_tensor.shape == loaded_tensor.shape:
            self.policy.policy_para_tensor.copy_(loaded_tensor)
            print("Updated policy paramter values successfully.")
        else:
            print("Error: Shape mismatch between existing tensor and loaded tensor.")
            
        episodic_return = torch.tensor([0],dtype=torch.float64,device=self.device) 
        for idx_t in trange(running_step):
            actions = self.policy.get_action(self.env.agent_state)
            rewards = self.env.step(actions)
            if torch.sum(rewards) != 1:
                breakpoint()
            episodic_return += self.para.gamma**(idx_t) * torch.sum(rewards)
        print(f"episodic_return is {episodic_return}")

def main():
    
    # initialize the execution env
    execute = Execution()
    
    # path of model
    path_of_model = "/home/dongmingwang/project/coupled_rl/results/final20241225-105743kappa_o_4_kappa_r_1/kappa_o_4_kappa_r_1_model.pt"
    
    # Extract the filename from the path
    filename = Path(path_of_model).name
    
    # Regular expression to extract integer values of kappa_o and kappa_r
    pattern = r"kappa_o_(\d+)_kappa_r_(\d+)"
    
    # Search for the pattern in the filename
    match = re.search(pattern, filename)

    if match:
        # Extract values as integers
        kappa_o = int(match.group(1))
        kappa_r = int(match.group(2))
        print(f"kappa_o: {kappa_o}, kappa_r: {kappa_r}")
        
        # in case of changing the parameters, update them when using new model
        execute.para.kappa_o = kappa_o
        execute.para.kappa_r = kappa_r
        
    else:
        print("Pattern not found in filename.")
        
    # running steps
    running_step = 1000
    # execute
    execute.execution(path_of_model,running_step)       

if __name__ == "__main__":
    main()
