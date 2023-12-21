import numpy as np
import gym
from buffer import ReplayBuffer
from dqn import DQN
import time
import matplotlib.pyplot as plt
import random
import os
import torch
import argparse

parser = argparse.ArgumentParser(description='Render environment.')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--untrained', action='store_true', default=False,
                    help='use untrained model')
args = parser.parse_args()

if args.untrained:
    model_name = 'untrained.pth'
else:
    model_name = 'ddqn.pth'
name = os.path.join(os.path.abspath(os.getcwd()), 'models', model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.render:
    env = gym.make("CartPole-v1", render_mode='human')
else:
    env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
model = DQN(state_size, action_size)
model.load_state_dict(torch.load(name))
done = False
state = env.reset()[0].reshape(1, state_size)
state = torch.from_numpy(state).float().to(device)
total_reward = 0
while not done:
    if args.render:
        env.render()
    q_values = model(state)
    action = torch.argmax(q_values).item()
    nextState, reward, done, x, _ = env.step(action)
    nextState = torch.from_numpy(nextState.reshape(1, state_size)).float().to(device)
    total_reward += reward
    state = nextState
print(f'Total reward: {total_reward}')
env.close()
