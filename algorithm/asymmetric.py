#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File   : asymmetric.py
@Author : Dongming Wang
@Email  : dongming.wang@email.ucr.edu
@Project: $(basename ~/project/coupled_rl/algorithm)
@Date   : 12/21/2024
@Time   : 15:30:58
@Info   : Description of the script
"""
import os
import torch
import time
import setproctitle

from tqdm import tqdm

class Asymmetric:
    def __init__(self, para, device):
        """
        Constructor for Asymmetric
        """
        ## Initialize class Asymmetric attributes here
        # parameters from parser
        self.device  = device
        self.agents_num = para.agents_num
        self.action_space_size = para.action_space_size
        self.state_space_size = para.state_space_size
        self.is_training = para.is_training
        
        
        # step size paramters 
        self.gamma = para.gamma
        self.h = para.h
        self.eta = para.eta
        self.t_0 = para.t_0
        self.kappa_o = para.kappa_o
        self.kappa_r = para.kappa_r
        self.steps_num = para.steps_num
        
        # initialize the Q talbe with all 0
        # potential limitation: in torch the dimension of all elements must be homogeneous, so it may be infreasible when dealing with inhomogeneous situation
        self.Q_table_init = torch.zeros(
            size=(self.agents_num,),
            device=device,
            dtype=torch.float64
        )
        # index to agents 
        for _ in range(2*self.kappa_o+1):
            shape = self.Q_table_init.shape
            new_shape = shape + (self.state_space_size,)
            self.Q_table_init = torch.zeros(new_shape, dtype=torch.float64, device=device)
        # index to actions
        for _ in range(2*self.kappa_o+1):
            shape = self.Q_table_init.shape
            new_shape = shape + (self.action_space_size,)
            self.Q_table_init = torch.zeros(new_shape, dtype=torch.float64, device=device)    
        
        # initialize the policy paramters as all 0 
        self.policy_para_tensor = torch.zeros(
            size=(self.agents_num, self.state_space_size, self.action_space_size),  # Shape of the tensor
            device=device,  # Device (CPU or CUDA) where the tensor is stored
            dtype=torch.float32  # Data type of the tensor elements (floating point numbers)
        )
        self.actions_init = torch.zeros(size=(self.agents_num,),device=device,dtype=bool)
        
        # initiallize the data collector to be a dictionary
        self.data_dict = {
            "state": torch.tensor([], dtype=torch.bool, device=device),  # Empty tensor for state
            "action": torch.tensor([], dtype=torch.bool, device=device),  # Empty tensor for action
            "reward": torch.tensor([], dtype=torch.bool, device=device)  # Empoty tensor for reward
        }           

        

    def save_state(self, state):
        
        # Clone and unsqueeze the state for saving
        state_for_save = state.clone().unsqueeze(0)

        # Check if self.data_dict["state"] is empty
        if self.data_dict["state"].numel() == 0:
            self.data_dict["state"] = state_for_save
        else:
            self.data_dict["state"] = torch.cat((self.data_dict["state"], state_for_save), dim=0)
            
    def save_action(self, action):
           
        # Clone and unsqueeze the action for saving
        action_for_save = action.clone().unsqueeze(0)

        # Check if self.data_dict["action"] is empty
        if self.data_dict["action"].numel() == 0:
            self.data_dict["action"] = action_for_save
        else:
            self.data_dict["action"] = torch.cat((self.data_dict["action"], action_for_save), dim=0)
            
        
    def save_reward(self, reward):
        
            
        # Clone and unsqueeze the reward for saving
        reward_for_save = reward.clone().unsqueeze(0)

        # Check if self.data_dict["reward"] is empty
        if self.data_dict["reward"].numel() == 0:
            self.data_dict["reward"] = reward_for_save
        else:
            self.data_dict["reward"] = torch.cat((self.data_dict["reward"], reward_for_save), dim=0)


    
    def clear_data_buffer(self):
        
        # Empty the data buffer of last episode
        self.data_dict = {
            "state": torch.tensor([], dtype=torch.bool, device=self.device),  # Empty tensor for state
            "action": torch.tensor([], dtype=torch.bool, device=self.device),  # Empty tensor for action
            "reward": torch.tensor([], dtype=torch.bool, device=self.device)  # Empoty tensor for reward
        }  
    
    
    def update_Q_table(self, step):
        alpha = self.h / (step - 1 + self.t_0)

        # get the state of time t-1
        state_pre = self.data_dict["state"][-2].clone()
        action_pre = self.data_dict["action"][-2].clone()
        reward_pre = self.data_dict["reward"][-2].clone()
        
        # get current state of time t
        state_current = self.data_dict["state"][-1].clone()
        action_current = self.data_dict["action"][-1].clone()
        
        # update the Q-table for the fist agent
        # assume there is a 0 agents before 0 with all state all zeros and action zeros
        for idx in range(self.agents_num):
            # make the state and action always be 0 for those pad visual agents
            # find the correct Q value to update
            # as the distributed setting, the agents can never over the range, so we do not need to concern than bother borders are overflow
            
            ### find the Q-value to update
            index_tensor_kappa_o = torch.empty((0,), dtype=torch.int32, device=self.device)
            if idx < self.kappa_o:
                # Collect indices to add in a list
                zero_pad_indices = [0] * (self.kappa_o - idx)  # List of zeros to append
                # Convert to a tensor
                pad_indices_tensor = torch.tensor(zero_pad_indices, dtype=torch.int32, device=self.device)
                # Concatenate the new tensor to the existing tensor
                index_tensor_kappa_o = torch.cat((index_tensor_kappa_o, pad_indices_tensor))

                # Handle reminders safely
                reminders = self.kappa_o + idx + 1
                index_copy = state_pre[:reminders]  # This remains a tensor
                # Concatenate directly
                index_tensor_kappa_o = torch.cat((index_tensor_kappa_o, index_copy))

                # The action part, in the same way
                zero_pad_indices = [0] * (self.kappa_o - idx)  # List of zeros to append
                # Convert to a tensor
                pad_indices_tensor = torch.tensor(zero_pad_indices, dtype=torch.int32, device=self.device)
                # Concatenate the new tensor to the existing tensor
                index_tensor_kappa_o = torch.cat((index_tensor_kappa_o, pad_indices_tensor))

                # Handle action reminders safely
                reminders = self.kappa_o + idx + 1
                index_copy = action_pre[:reminders]  # This remains a tensor
                # Concatenate directly
                index_tensor_kappa_o = torch.cat((index_tensor_kappa_o, index_copy))
    
            elif idx + self.kappa_o >= self.agents_num:
                
                index_copy = state_pre[idx-self.kappa_o:]
                # Concatenate directly
                index_tensor_kappa_o = torch.cat((index_tensor_kappa_o, index_copy))
                
                zero_pad_indices = [0] * (self.kappa_o + idx + 1 - self.agents_num)
                # Convert to a tensor
                pad_indices_tensor = torch.tensor(zero_pad_indices, dtype=torch.int32, device=self.device)
                # Concatenate the new tensor to the existing tensor
                index_tensor_kappa_o = torch.cat((index_tensor_kappa_o, pad_indices_tensor))
                
                
                
                index_copy = action_pre[idx-self.kappa_o:]
                # Concatenate directly
                index_tensor_kappa_o = torch.cat((index_tensor_kappa_o, index_copy))
                
                
                zero_pad_indices = [0] * (self.kappa_o + idx + 1 - self.agents_num)
                # Convert to a tensor
                pad_indices_tensor = torch.tensor(zero_pad_indices, dtype=torch.int32, device=self.device)
                # Concatenate the new tensor to the existing tensor
                index_tensor_kappa_o = torch.cat((index_tensor_kappa_o, pad_indices_tensor))
            
            else:
                index_copy = state_pre[idx-self.kappa_o:idx+self.kappa_o+1]
                index_tensor_kappa_o = torch.cat((index_tensor_kappa_o, index_copy))
                
                index_copy = action_pre[idx-self.kappa_o:idx+self.kappa_o+1]
                index_tensor_kappa_o = torch.cat((index_tensor_kappa_o, index_copy))
        
            ### find the Q-value used for update
            after_index_tensor_kappa_o = torch.empty((0,), dtype=torch.int32, device=self.device)
            if idx < self.kappa_o:
                # Collect indices to add in a list
                zero_pad_indices = [0] * (self.kappa_o - idx)  # List of zeros to append
                # Convert to a tensor
                pad_indices_tensor = torch.tensor(zero_pad_indices, dtype=torch.int32, device=self.device)
                # Concatenate the new tensor to the existing tensor
                after_index_tensor_kappa_o = torch.cat((after_index_tensor_kappa_o, pad_indices_tensor))

                # Handle reminders safely
                reminders = self.kappa_o + idx + 1
                index_copy = state_current[:reminders]  # This remains a tensor
                # Concatenate directly
                after_index_tensor_kappa_o = torch.cat((after_index_tensor_kappa_o, index_copy))

                # The action part, in the same way
                zero_pad_indices = [0] * (self.kappa_o - idx)  # List of zeros to append
                # Convert to a tensor
                pad_indices_tensor = torch.tensor(zero_pad_indices, dtype=torch.int32, device=self.device)
                # Concatenate the new tensor to the existing tensor
                after_index_tensor_kappa_o = torch.cat((after_index_tensor_kappa_o, pad_indices_tensor))

                # Handle action reminders safely
                reminders = self.kappa_o + idx + 1
                index_copy = action_current[:reminders]  # This remains a tensor
                # Concatenate directly
                after_index_tensor_kappa_o = torch.cat((after_index_tensor_kappa_o, index_copy))
    
            elif idx + self.kappa_o >= self.agents_num:
                
                index_copy = state_current[idx-self.kappa_o:]
                # Concatenate directly
                after_index_tensor_kappa_o = torch.cat((after_index_tensor_kappa_o, index_copy))
                
                zero_pad_indices = [0] * (self.kappa_o + idx + 1 - self.agents_num)
                # Convert to a tensor
                pad_indices_tensor = torch.tensor(zero_pad_indices, dtype=torch.int32, device=self.device)
                # Concatenate the new tensor to the existing tensor
                after_index_tensor_kappa_o = torch.cat((after_index_tensor_kappa_o, pad_indices_tensor))
                
                
                
                index_copy = action_current[idx-self.kappa_o:]
                # Concatenate directly
                after_index_tensor_kappa_o = torch.cat((after_index_tensor_kappa_o, index_copy))
                
                
                zero_pad_indices = [0] * (self.kappa_o + idx + 1 - self.agents_num)
                # Convert to a tensor
                pad_indices_tensor = torch.tensor(zero_pad_indices, dtype=torch.int32, device=self.device)
                # Concatenate the new tensor to the existing tensor
                after_index_tensor_kappa_o = torch.cat((after_index_tensor_kappa_o, pad_indices_tensor))
            
            else:
                index_copy = state_current[idx-self.kappa_o:idx+self.kappa_o+1]
                after_index_tensor_kappa_o = torch.cat((after_index_tensor_kappa_o, index_copy))
                
                index_copy = action_current[idx-self.kappa_o:idx+self.kappa_o+1]
                after_index_tensor_kappa_o = torch.cat((after_index_tensor_kappa_o, index_copy))
            
            # incase of be in the same state and cause multiple update, we clone the origin Q tale
            Q_table_pre = self.Q_table_init.clone()
            
            
            # calculate the reward in advance
            reward_term = torch.tensor(0, dtype=torch.float64,device=self.device)
            for idx_r in range(idx-self.kappa_r,idx+self.kappa_r+1):
                if idx_r < 0 or idx_r+1>self.agents_num:
                    reward_term += 0
                else:
                    reward_term += reward_pre[idx_r]
                    
            # Convert idx to a tensor with one element
            idx_tensor = torch.tensor([idx], device=self.device)  # Convert idx to a tensor of shape (1,)

            # Concatenate the tensors
            indices_to_update = tuple(torch.cat((idx_tensor, index_tensor_kappa_o)))
            
            indices_used_for_update = tuple(torch.cat((idx_tensor,after_index_tensor_kappa_o)))
            
            self.Q_table_init[indices_to_update] = (1 - alpha) * Q_table_pre[indices_to_update] + alpha * \
                (1 / self.agents_num * reward_term + self.gamma * Q_table_pre[indices_used_for_update])
            
            
            

            
                
    
    def update_policy(self):
        pass
    
    # get actions use softmax when training and greedy when execution, contains the following three functions
    def get_action(self, agents_state):
        
        if self.is_training:
            return self.softmax_policy(agents_state)
        else:
            return self.greedy_policy(agents_state)
        
    
    def softmax_policy(self,agents_state):
        
        actions = self.actions_init.clone()
        
        for i in range(self.agents_num):
            policy_logits = self.policy_para_tensor[i]
            probabilities = torch.softmax(policy_logits, dim=-1)
            action = torch.multinomial(probabilities[int(agents_state[i])], num_samples=1).item()  # Sample action for state
            actions[i] = bool(action)
        return actions

            
            
    
    def greedy_policy(self,agents_state):
        actions = self.actions_init.clone()
        
        for i in range(self.agents_num):
            greedy_action = torch.argmax(agents_state[i].to(torch.int)).item()
            actions[i] = greedy_action
        return actions

    
    
    def save_model(self, save_path):
        # Get the current runtime (time elapsed since epoch)
        run_time = time.strftime("%Y%m%d-%H%M%S")
        
        # Set the process title (for display in system)
        setproctitle.setproctitle(f"training-{run_time}")
        
        # Create folder path with "results" and the run time
        folder_path = os.path.join(save_path, "final" + run_time)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Define the full path to the file, including the filename
        file_path = os.path.join(folder_path, "model.pt")
        
        # Save the tensor to the file path
        torch.save(self.policy_para_tensor, file_path)

        tqdm.write(f"Model saved to {file_path}")
