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

class SyntheticEnv():
    def __init__(self, para, device, *args, **kwargs):
        """
        Initialize the SyntheticEnv():.
        """
        # generally used parameters
        self.device = device
        
        # agents parameters
        self.agent_num = para.agents_num
        
        # initialize the start state to be all true
        self.agent_state = torch.ones(size=(self.agent_num, ), device=device, dtype=torch.bool)
        
        # initialize the initial reward to be all 0
        self.reward = torch.zeros(size=(self.agent_num, ), device=device)
        
        # transition model parameters
        self.trans_prob = para.trans_prob
    
    def step(self,actions):
        assert self.agent_state.device == actions.device, \
            f"Device mismatch: state device {self.agent_state.device} vs action device {actions.device}"
            
        assert self.agent_state.size() == actions.size(), \
            f"Action sizes mismatch: state size {self.agent_state.size()} vs action size {actions.size()}"
        
        # avoiding multiple update
        agent_state_pre = self.agent_state.clone()
        
        # update the state of agent 1
        if agent_state_pre[1]:
            self.agent_state[0] = True
        else:
            self.agent_state[0] = False
        
        # update the state of agent 2-n-1
        for idx in range(1,self.agent_num-1):
            
            # case 1: when s_{i+1} = 1, a_i = 1, the s_i(t+1) = 1
            
            if agent_state_pre[idx+1] and actions[idx]:
                self.agent_state[idx] = True
            elif not agent_state_pre[idx+1] and actions[idx]:
                self.agent_state[idx] = self.stochastic_state()
            else:
                self.agent_state[idx] = False
                
        # update the state of agent n
        
        if actions[-1]:
            self.agent_state[-1] = True
        else:
            self.agent_state[-1] = False
        
        # new_state = self.agent_state
        reward = self.get_reward()
        return reward
            
        
            
        
    def get_reward(self):
        
        # reward r_1 = 1 when s_1 = 1, and 0 otherwise
        reward = self.reward.clone()
        if self.agent_state[0]:
            reward[0] += 1

        return reward
            
        
        
    # generate stochastic state
    def stochastic_state(self):
        return torch.bernoulli(torch.full((1,), self.trans_prob, device=self.device))
        
    def reset(self):
        ### reset the state to be initial state
        # initialize the start state to be all true
        self.agent_state = torch.ones(size=(self.agent_num, ), device=self.device, dtype=torch.bool)