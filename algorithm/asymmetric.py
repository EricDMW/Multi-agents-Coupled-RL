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
import matplotlib
import setproctitle

import matplotlib.pyplot as plt

from tqdm import tqdm


class Asymmetric:
    def __init__(self, para, device, *args, **kwargs):
        """
        Constructor for Asymmetric
        """
        # matplotlib.use('Agg')  # Non-interactive backend
        # matplotlib.use('TkAgg')  
        # matplotlib.use('QtAgg')
        
        
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
        
        ### initialize the Q talbe with all 0 Q(i,s_{N_i},a_{N_i})
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
            "reward": torch.tensor([], dtype=torch.bool, device=device)  # Empty tensor for reward
        }           

        # initialize the gradient saver to be a 2-d tensor
        self.episodic_gradient_record = torch.empty(0,device=device)
        

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
            "reward": torch.tensor([], dtype=torch.bool, device=self.device)  # Empty tensor for reward
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
            indices_to_update = self.get_Q_indices(idx,state_pre,action_pre)
           
            ### find the Q-value used for update
            indices_used_for_update = self.get_Q_indices(idx,state_current,action_current)
                        
            # incase of be in the same state and cause multiple update, we clone the origin Q tale
            Q_table_pre = self.Q_table_init.clone()
            
            
            # calculate the reward in advance
            reward_term = torch.tensor(0, dtype=torch.float64,device=self.device)
            for idx_r in range(idx-self.kappa_r,idx+self.kappa_r+1):
                if idx_r < 0 or idx_r+1>self.agents_num:
                    reward_term += 0
                else:
                    reward_term += reward_pre[idx_r]
                    
            
            
            self.Q_table_init[indices_to_update] = (1 - alpha) * Q_table_pre[indices_to_update] + alpha * \
                (1 / self.agents_num * reward_term + self.gamma * Q_table_pre[indices_used_for_update])
            
    def get_Q_indices(self, idx, state_tensor,action_tensor):
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
            index_copy = state_tensor[:reminders]  # This remains a tensor
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
            index_copy = action_tensor[:reminders]  # This remains a tensor
            # Concatenate directly
            index_tensor_kappa_o = torch.cat((index_tensor_kappa_o, index_copy))

        elif idx + self.kappa_o >= self.agents_num:
            
            index_copy = state_tensor[idx-self.kappa_o:]
            # Concatenate directly
            index_tensor_kappa_o = torch.cat((index_tensor_kappa_o, index_copy))
            
            zero_pad_indices = [0] * (self.kappa_o + idx + 1 - self.agents_num)
            # Convert to a tensor
            pad_indices_tensor = torch.tensor(zero_pad_indices, dtype=torch.int32, device=self.device)
            # Concatenate the new tensor to the existing tensor
            index_tensor_kappa_o = torch.cat((index_tensor_kappa_o, pad_indices_tensor))
            
            
            
            index_copy = action_tensor[idx-self.kappa_o:]
            # Concatenate directly
            index_tensor_kappa_o = torch.cat((index_tensor_kappa_o, index_copy))
            
                        
            zero_pad_indices = [0] * (self.kappa_o + idx + 1 - self.agents_num)
            # Convert to a tensor
            pad_indices_tensor = torch.tensor(zero_pad_indices, dtype=torch.int32, device=self.device)
            # Concatenate the new tensor to the existing tensor
            index_tensor_kappa_o = torch.cat((index_tensor_kappa_o, pad_indices_tensor))
        
        else:
            index_copy = state_tensor[idx-self.kappa_o:idx+self.kappa_o+1]
            index_tensor_kappa_o = torch.cat((index_tensor_kappa_o, index_copy))
            
            index_copy = action_tensor[idx-self.kappa_o:idx+self.kappa_o+1]
            index_tensor_kappa_o = torch.cat((index_tensor_kappa_o, index_copy))
        # Convert idx to a tensor with one element
        idx_tensor = torch.tensor([idx], device=self.device)  # Convert idx to a tensor of shape (1,)

        # Concatenate the tensors
        indices_in_Q_tensor = tuple(torch.cat((idx_tensor, index_tensor_kappa_o)))    
        
        return indices_in_Q_tensor
            
        

        
                
    
    def update_policy(self,episode):
        
        ### check the data collection, in case of wrong info
        assert self.steps_num == self.data_dict["state"].size(0), (
            f"Mismatch in step count: expected {self.steps_num}, but found {self.data_dict["state"]} in data_dict['state']."
        )
        
        assert self.steps_num == self.data_dict["action"].size(0), (
            f"Mismatch in step count: expected {self.steps_num}, but found {self.data_dict["state"]} in data_dict['state']."
        )
        
        assert self.steps_num == self.data_dict["reward"].size(0), (
            f"Mismatch in step count: expected {self.steps_num}, but found {self.data_dict["state"]} in data_dict['state']."
        )
        
        ###
        
        eta_m = self.eta / torch.sqrt(torch.tensor(episode + 1, dtype=torch.float32, device=self.device))

        # All zero tensor used for saving the gradient information
        gradient_saver_at_episode = torch.zeros(self.agents_num,device=self.device)
        
        ### update the the paramters theta, self.policy_para_tensor
        for idx_step in range(self.steps_num):
            # access to the states and actions
            global_state = self.data_dict["state"][idx_step]
            global_action = self.data_dict["action"][idx_step]
            # update the parameters for each agents
            for idx in range(self.agents_num):
                
                # get the currespond Q value in the Q table
                Q_index = self.get_Q_indices(idx,global_state, global_action)
                
                # get the policy paramters to be updated
                theta_update_index = tuple([idx,int(global_state[idx])])
                
                # avoid potential self update error
                policy_para_temp = self.policy_para_tensor[theta_update_index].clone()
                policy_para_temp.requires_grad_()  # Set requires_grad to True
                
                ### update the corresponding parameters one by one
                # gradient term
                pi_for_update = torch.softmax(policy_para_temp, dim=0)
                
                # log pi
                log_pi_for_update = torch.log(pi_for_update[int(global_action[idx])])
                
                # nabla log pi
                log_pi_for_update.backward()
                
                # update the policy
                self.policy_para_tensor[theta_update_index] += eta_m * self.gamma**idx_step * self.Q_table_init[Q_index] * policy_para_temp.grad
                
                # save the gradient to tensor
                gradient_saver_at_episode[idx] += torch.norm(self.gamma**idx_step * self.Q_table_init[Q_index] * policy_para_temp.grad)
        
        
        # append current 
        self.episodic_gradient_record  = eta_m * torch.cat([self.episodic_gradient_record,gradient_saver_at_episode.unsqueeze(0)],dim=0)        
                
    
    
    
    
    # get actions use softmax when training and greedy when execution, contains the following three functions
    def get_action(self, agents_state):
        
        if self.is_training:
            return self.softmax_policy(agents_state)
        else:
            return self.greedy_policy(agents_state)
        
    
    def softmax_policy(self,agents_state):
        
        # initialize an action
        actions = self.actions_init.clone()
        
        for i in range(self.agents_num):
            policy_logits = self.policy_para_tensor[i]
            probabilities = torch.softmax(policy_logits, dim=-1)
            action = torch.multinomial(probabilities[int(agents_state[i])], num_samples=1).item()  # Sample action for state
            actions[i] = bool(action)
        return actions

            
            
    
    def greedy_policy(self,agents_state):
        
        # Initialize an action
        actions = self.actions_init.clone()

        for i in range(self.agents_num):
            policy_logits = self.policy_para_tensor[i]
            
            # Get the logits for the current agent's state
            logits_for_state = policy_logits[agents_state[i].to(torch.int)]
            
            # Reverse the order of logits to prioritize higher indices for ties
            reversed_logits = logits_for_state.flip(dims=[0])
            
            # Get the greedy action with tie-breaking by larger index
            greedy_action = torch.argmax(reversed_logits).item()
            
            # Correct the index because we flipped the logits
            greedy_action = len(logits_for_state) - 1 - greedy_action
            
            actions[i] = bool(greedy_action)

        return actions

    
    def plot_episodic_return(self,save_path,episodic_return_save,episode,if_final=False):
        
        # Get the current runtime (time elapsed since epoch)
        run_time = time.strftime("%Y%m%d-%H%M%S")
        
        # Set the process title (for display in system)
        setproctitle.setproctitle(f"training-{run_time}")
        
        # Create folder path with "results" and the run time
        if if_final:
            folder_path = os.path.join(save_path, "final" + run_time)
        else:
            folder_path = save_path
        
            
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Define the full path to the file, including the filename
        file_path = os.path.join(folder_path, f"episodic_{episode}_return_plot_{run_time}.png")
        
        # Assuming episodic_return_save is a tensor on CUDA (GPU)
        episodic_return_save = episodic_return_save.cpu()  # Move tensor to CPU

        # Convert the tensor to a NumPy array
        episodic_return_save  = episodic_return_save.squeeze().numpy()  # Squeeze removes the singleton dimension

        
        # Plot the data
        plt.plot(episodic_return_save)
        plt.xlabel('Episodes')  # Label for the x-axis
        plt.ylabel('Return')  # Label for the y-axis
        plt.title('Epicodic Return')  # Title of the plot
        plt.savefig(file_path)
        plt.close()

        # Display the plot
        # plt.show()
        

        tqdm.write(f"Plot saved to: {file_path}")
                        
    
    
    def save_model(self, save_path, episodic_return_save):
        # Get the current runtime (time elapsed since epoch)
        run_time = time.strftime("%Y%m%d-%H%M%S")
        
        # Set the process title (for display in system)
        setproctitle.setproctitle(f"training-{run_time}")
        
        # Create folder path with "results" and the run time
        folder_path = os.path.join(save_path, "final" + run_time + f"kappa_o_{self.kappa_o}_kappa_r_{self.kappa_r}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Define the full path to the file, including the filename
        file_path = os.path.join(folder_path, f"kappa_o_{self.kappa_o}_kappa_r_{self.kappa_r}_model.pt")
        
        # Save the tensor to the file path
        torch.save(self.policy_para_tensor, file_path)
        
        file_path = os.path.join(folder_path, "episodic_return.pt")
        
        # Save the tensor to the file path
        torch.save(episodic_return_save, file_path)

        file_path = os.path.join(folder_path, "gradient_over_agents_return.pt")
        torch.save(self.episodic_gradient_record, file_path)
        
        
        tqdm.write(f"Model saved to {file_path}")
        