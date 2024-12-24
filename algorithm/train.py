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
import time

from pathlib import Path
from tqdm import tqdm, trange

# avoid potential import error
parents_path = Path(__file__).parents[1]
sys.path.append(str(parents_path))

current_level_path = Path(__file__).parents[0]
sys.path.append(str(current_level_path))


# parameters 
from config import get_config
from utils import get_device, set_seed

from asymmetric import Asymmetric
from env import SyntheticEnv

### Training preparation: parameters and device setting
def initialize_para():
     
    # set the running device
    device = get_device()
    
    # parser parameters
    parser = get_config()
    para = parser.parse_args()
    
    return device, para

# Function to set the folder name dynamically using the runtime
def create_results_folder():
    # Define the main folder path
    main_folder = "results"
    
    # Check if the main folder exists, if not, create it
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    
    # Get the current runtime (time elapsed since epoch)
    run_time = time.strftime("%Y%m%d-%H%M%S")
    
    # Set the process title (for display in system)
    setproctitle.setproctitle(f"training-{run_time}")
    
    # Create folder path with "results" and the run time
    folder_path = os.path.join(main_folder, run_time)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    return folder_path, main_folder

def train():
    """
    Main function, used for training.
    """
    ### Training preparation: parameters and device setting
    
    # set the running device
    device, para = initialize_para()
    
    ### Choose to set the seed or not
    # set the seed
    if para.fix_seed:
        try:
            set_seed(para.rand_seed)
            # print(f"Seed set to {para.rand_seed} for reproducibility.")
        except ValueError as e:
            print(f"Error: {e}")
    else:
        print("Seed is not fixed.")
    
    ### buid the environment
    
    # initialize the policy
    policy = Asymmetric(para,device)
    
    # initialize the environment
    env = SyntheticEnv(para,device)
    
    
    ### Create folder for saving the results
    folder_path, main_folder = create_results_folder()
  
    
    ### Used for record variables
    episodic_return_save = torch.empty(0, dtype=torch.float64,device=device)
    
    # training progress
    for episode in trange(para.episode_num):
        # reset the enrironment
        env.reset()
        
        # empty the data saver when start a new episode
        policy.clear_data_buffer()
        

        # used for calculate episodic return
        episodic_return = torch.tensor([0],dtype=torch.float32,device=device)
        
        for step in range(para.steps_num):
            
            # save the state of current env
            policy.save_state(env.agent_state)
            
            actions = policy.get_action(env.agent_state)
            
            # save actions correspond to env
            policy.save_action(actions)

            reward = env.step(actions)
            
            # save reward correspond to env
            policy.save_reward(reward)
            
            # calculate episodic return
            episodic_return += policy.gamma**step * torch.sum(reward)
            
            
            
            if step > 0:
                # update Q when t>0
                policy.update_Q_table(step)
                
        # record episodic return
        episodic_return_save = torch.cat((episodic_return_save, episodic_return), dim=0)

        policy.update_policy(episode)
        
        
            
        # save the procedure model incase of loss connection    
        if (episode + 1) % para.save_frequency == 0:
            episodic_save_path = os.path.join(folder_path, f"episode_{episode+1}_kappa_o_{policy.kappa_o}_kappa_r_{policy.kappa_r}_tensor.pt")
            
            policy.plot_episodic_return(folder_path, episodic_return_save, episode)
            torch.save(policy.policy_para_tensor, episodic_save_path)
            tqdm.write(f"Saved tensor for episode {episode+1} to {episodic_save_path}")    
    
    
    policy.save_model(main_folder,episodic_return_save)
    policy.plot_episodic_return(main_folder,episodic_return_save, episode=-1, if_final=True)
    
    
    

if __name__ == "__main__":
    train()