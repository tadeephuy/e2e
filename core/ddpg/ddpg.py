import multiprocessing as mp
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torchvision import transforms
from . import Critic, Actor, Memory

class Agent:
    """
    DDPG Agent class
    
    Arguments:
        critic_arch: cnn feature extractor for critic network
        actor_arch: cnn feature extractor for actor network
        critic_hidden_size: # channels for feature maps from critic_arch
        actor_hidden_size: # channels for feature maps from actor_arch
        critic_learning_rate: learning rate for critic network
        actor_learning_rate: learning rate for actor network
        gamma: discount factor
        tau: update rate for target networks
    """
    def __init__(self,
                critic_arch, actor_arch,
                critic_hidden_size, actor_hidden_size,
                action_size,
                memory_size,
                critic_learning_rate=1e-3,
                actor_learning_rate=1e-3,
                gamma=0.99, tau=1e-2, device=torch.device('cuda:0')):
        
        self.device = device
        self.gamma, self.tau = gamma, tau

        self.critic = Critic(arch=critic_arch,
                             hidden_size=critic_hidden_size,
                             action_size=action_size).to(self.device)
        self.actor = Actor(arch=actor_arch,
                           hidden_size=actor_hidden_size,
                           action_size=action_size).to(self.device)
        self.critic_target = Critic(arch=critic_arch,
                             hidden_size=critic_hidden_size,
                             action_size=action_size).to(self.device)
        self.actor_target = Actor(arch=actor_arch,
                                  hidden_size=actor_hidden_size,
                                  action_size=action_size).to(self.device)

        # initialize target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.memory = Memory(memory_size)        
        self.critic_criterion  = nn.MSELoss().to(self.device)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
      
    def preprocess_state(self, state):
        def ToTensor3Channels(x):
            x = torch.from_numpy(x.astype(np.float32))
            x[...,:3] = x[...,:3]/255
            x = x.permute(2, 0, 1)
            return x

        preprocess = transforms.Compose([
            transforms.Lambda(ToTensor3Channels),
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0],
                                 std=[0.229, 0.224, 0.225, 1])
        ])
        state = preprocess(state).unsqueeze(0)
        state = Variable(state)
        return state


    def get_action(self, state):
        state = self.preprocess_state(state)
        action = self.actor.forward(state.to(self.device))
        action = action.detach().cpu().numpy()[0]
        
        return action

    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)

        for i in range(batch_size):
            states[i] = self.preprocess_state(states[i])
            next_states[i] = self.preprocess_state(next_states[i])
        states = torch.cat(states, axis=0)
        next_states = torch.cat(next_states, axis=0)
#         import pdb; pdb.set_trace()
        states, actions, rewards, next_states, _ = map(torch.FloatTensor, [states, actions, rewards, next_states, _])
        

        states, actions, rewards, next_states = states.to(self.device), actions.to(self.device), rewards.to(self.device), next_states.to(self.device)
        # Critic loss
        Qvals = self.critic.forward(states, actions)
        
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions)
        Qprime = rewards + self.gamma*next_Q
        
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        this_actions = self.actor.forward(states)
        policy_loss = - self.critic.forward(states, this_actions).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data*self.tau + target_param.data*(1.0 - self.tau))
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data*self.tau + target_param.data*(1.0 - self.tau))

                                          
