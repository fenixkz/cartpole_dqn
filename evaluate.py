import numpy as np
import gym
from buffer import ReplayBuffer
from dqn import DQN
import time
import matplotlib.pyplot as plt
import random
import os
from tensorflow import keras
import argparse 

parser = argparse.ArgumentParser(description='Render environment.')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--untrained', action='store_true', default=False,
                    help='use untrained model')
args = parser.parse_args()

if args.untrained:
    model_name = 'dqn_untrained.h5'
else:
    model_name = 'dqn.h5'
name = os.path.join(os.path.abspath(os.getcwd()), 'models', model_name)
model = keras.models.load_model(name)

if args.render:
    env = gym.make("CartPole-v1", render_mode='human')
else:
    env = gym.make("CartPole-v1")
done = False
state = env.reset()[0].reshape(1,4)
total_reward = 0
while not done:
    if args.render:
        env.render()
    action = np.argmax(model.predict(state, verbose=0))
    nextState, reward, done, x, _ = env.step(action)
    nextState = nextState.reshape(1, 4)
    total_reward += reward
    state = nextState
print(f'Total reward: {total_reward}')
env.close()