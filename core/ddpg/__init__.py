import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import torch
from torch import nn
import numpy as np
import random
from collections import deque

class Critic(nn.Module):
    """
    Critic Network wrapper
    Arguments:
        arch (nn.Module):  cnn feature extractor
        hidden_size (int): # channels of the feature map
        action_size (int): # dim of action
    """
    def __init__(self, arch, hidden_size, action_size):
        super(Critic, self).__init__()
        self.transistion = nn.Conv2d(4, 3, 1) # from 4 channels to 3  channels
        self.core = arch
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Linear(hidden_size + action_size, hidden_size), nn.ReLU(),
                                  nn.Linear(hidden_size, 1))

    def forward(self, state, action):
        state = self.transistion(state)
        state = self.core(state)
        state = self.pool(state)
        state = nn.Flatten()(state)
        x = torch.cat([state, action], dim=1)
        return self.head(x)
        

class Actor(nn.Module):
    """
    Actor Network wrapper
    Arguments:
        arch (nn.Module):  cnn feature extractor
        hidden_size (int): # channels of the feature map
        action_size (int): # dim of action
    """
    def __init__(self, arch, hidden_size, action_size):
        super(Actor, self).__init__()
        self.transistion = nn.Conv2d(4, 3, 1) # from 4 channels to 3  channels
        self.core = arch
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                  nn.Linear(hidden_size, action_size))

    def forward(self, state):
        x = self.transistion(state)
        x = self.core(x)
        x = self.pool(x)
        x = self.head(x)
        return x.sigmoid()

class Memory:
    """
    Replay Buffer
    
    Arguments:
        max_size: length of the memory
    """
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch, action_batch, reward_batch, \
             next_state_batch, done_batch = [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)
        
        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        return state_batch, action_batch, reward_batch,\
            next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space # action_space.shape[0]
        self.low          = 0 #action_space.low
        self.high         = 1 # action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0): 
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

from .ddpg import Agent



        
