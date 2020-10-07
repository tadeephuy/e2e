import sys
import numpy as np
import pandas as pd
import torch
from ..core.ddpg import Agent, OUNoise
from ..core.environment import Environment


# Define arguments

CRITIC_ARCH = None
ACTOR_ARCH = None
CRITIC_SIZE = None
ACTOR_SIZE = None
ACTION_SIZE = None
CRITIC_LR = None
ACTOR_LR = None
GAMMA = None
TAU = None

IMG_DIR = None
IMG_SIZE = 320
BLOB_SIZE = 5
DONE_THRESHOLD = 0.5

BATCH_SIZE = 64
N_EPISODE = 50
MAX_LENGTH_EPISODE = 20

# Define variables

agent = Agent(
    critic_arch=CRITIC_ARCH, 
    actor_arch=ACTOR_ARCH, 
    critic_hidden_size=CRITIC_SIZE, 
    actor_hidden_size=ACTOR_SIZE, 
    action_size=ACTION_SIZE,
    critic_learning_rate=CRITIC_LR,
    actor_learning_rate=ACTOR_LR,
    gamma=GAMMA,
    tau=TAU
    )

env = Environment(
    agent=agent,
    classifier=classifier,
    img_dir=IMG_DIR,
    img_size=IMG_SIZE,
    blob_size=BLOB_SIZE,
    done_threshold=DONE_THRESHOLD
    )

noise = OUNoise(ACTION_SIZE)

batch_size = BATCH_SIZE
rewards = []
avg_rewards = []

for episode in range(N_EPISODE):
    state = env.reset()
    noise.reset()
    episode_reward = 0

    for step in range(MAX_LENGTH_EPISODE):
        action = agent.get_action(state)
        action = noise.get_action(action=action, t=step)
        new_state, reward, done, info = env.step(action)
        agent.memory.push(state, action, reward, new_state, done)

        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        state = new_state
        episode_reward += reward

        if done:
            print(f"Done episode {episode:<3} with reward of: {episode_reward:.2f}, avg reward: {np.mean(rewards)}")
            break
    
    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards))

torch.save(agent.critic.state_dict(), 'critic.pth')
torch.save(agent.actor.state_dict(), 'actor.pth')
torch.save(agent.critic_target.state_dict(), 'critic_target.pth')
torch.save(agent.actor_target.state_dict(), 'actor_target.pth')
