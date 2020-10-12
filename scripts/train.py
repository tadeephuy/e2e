import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from fastprogress import progress_bar
from core.ddpg import Agent, OUNoise
from core.environment import Environment

class Trainer:
    def __init__(self, config, critic_arch, actor_arch, classifier):
        self.config = config
        self.critic_arch, self.actor_arch, self.classifier = critic_arch, actor_arch, classifier

        self.initialize(self.config)
    
    def initialize(self, config):
        self.config = config

        # initialize agent
        self.agent = Agent(
            critic_arch         = self.critic_arch, 
            actor_arch          = self.actor_arch, 
            critic_hidden_size  = self.config['CRITIC_SIZE'], 
            actor_hidden_size   = self.config['ACTOR_SIZE'], 
            action_size         = self.config['ACTION_SIZE'],
            memory_size         = self.config['MEMORY_SIZE'],
            critic_learning_rate= self.config['CRITIC_LR'],
            actor_learning_rate = self.config['ACTOR_LR'],
            gamma               = self.config['GAMMA'],
            tau                 = self.config['TAU']
            )

        # intialize environment
        self.env = Environment(
            agent               = self.agent,
            classifier          = self.classifier,
            img_dir             = self.config['IMG_PATH'],
            img_size            = self.config['IMG_SIZE'],
            blob_size           = self.config['BLOB_SIZE'],
            done_threshold      = self.config['DONE_THRESHOLD']
            )
        
        self.noise = OUNoise(self.config['ACTION_SIZE'])

    def run(self):
        batch_size = self.config['BATCH_SIZE']
        rewards = []
        avg_rewards = []

        for episode in range(self.config['N_EPISODES']):
            state = self.env.reset()
            self.noise.reset()
            episode_reward = 0
            
            bar = progress_bar(range(self.config['MAX_LENGTH_EPISODE']))
            for step in bar:
                action = self.agent.get_action(state)
                action = self.noise.get_action(action=action, t=step)
                new_state, reward, done, info = self.env.step(action)
                self.agent.memory.push(state, action, reward, new_state, done)

                if len(self.agent.memory) > batch_size:
                    self.agent.update(batch_size)

                state = new_state
                episode_reward += reward
                
                bar.comment = f"reward: {reward:.3f} - episode_reward: {episode_reward:.3f}"

                if done:
                    print(f"Done episode {episode:<3} with reward of: {episode_reward:.2f}, avg reward: {np.mean(rewards)}")
                    break

            rewards.append(episode_reward)
            avg_rewards.append(np.mean(rewards))
        return rewards, avg_rewards

    def save(self, name=None):
        """
        specify `name` to override experiment name in config
        """
        name = name or self.config['NAME']
        state_dict = {
            'critic': self.agent.critic.state_dict(),
            'actor': self.agent.actor.state_dict(),
            'critic_target': self.agent.critic_target.state_dict(),
            'actor_target': self.agent.actor_target.state_dict()
        }

        torch.save(state_dict, f"{self.config['WEIGHT_PATH']}/{name}.pth")

    def load(self, name=None):
        """
        specify `name` to override experiment name in config
        """
        name = name or self.config['NAME']
        state_dict = torch.load(f"{self.config['WEIGHT_PATH']}/{name}.pth")

        self.agent.critic.load_state_dict(state_dict['critic'])
        self.agent.actor.load_state_dict(state_dict['actor'])
        self.agent.critic_target.load_state_dict(state_dict['critic_target'])
        self.agent.actor_target.load_state_dict(state_dict['actor_target'])
        