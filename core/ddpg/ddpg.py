import torch
from torch.autograd import Variable
from torch import optim
from . import Critic, Actor, Memory

class Agent:
    def __init__(self,
                critic_arch, actor_arch,
                hidden_size, action_size,
                critic_learning_rate=1e-3,
                actor_learning_rate=1e-3,
                gamma=0.99, tau=1e-2):
        self.critic = Critic(arch=critic_arch,
                             hidden_size=hidden_size,
                             action_size=action_size)
        self.actor = Actor(arch=actor_arch, action_size=action_size)
        self.critic_target = Critic(arch=critic_arch,
                             hidden_size=hidden_size,
                             action_size=action_size)
        self.actor_target = Actor(arch=actor_arch, action_size=action_size)

        # initialize target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.memory = Memory(max_memory_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0,0]
        return action

    def update(self, batch_size):
        states, actions, rewards, next_states, _ = map(torch.FloatTensor, self.memory.sample(batch_size))
        
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

                                          