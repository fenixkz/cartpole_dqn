import numpy as np
import gym
from buffer import ReplayBuffer
from dqn import DQN
import time
import matplotlib.pyplot as plt
import random
import os
from tensorflow import keras

path = os.path.abspath(os.getcwd())
name = str(path) + "/dqn.h5"
model = keras.models.load_model(name)

env = gym.make("CartPole-v1", render_mode='human')
done = False
state = env.reset()[0].reshape(1,4)
total_reward = 0
while not done:
    env.render()
    action = np.argmax(model.predict(state))
    nextState, reward, done, x, _ = env.step(action)
    nextState = nextState.reshape(1, 4)
    total_reward += reward
    state = nextState
print(total_reward)
