from dqn import DQN
import numpy as np
import gym
import time
import os
import keras
env = gym.make("CartPole-v1")
policy_net = DQN(env.observation_space.shape[0], env.action_space.n)
state = env.reset().reshape(-1,4)
# state = np.reshape(state, [1, 4])
q = policy_net.model.predict(state)[0]
a = np.argmax(q)
print(q)
print(a)
print(np.shape(q))
print(q[a])
print(np.max(policy_net.model.predict(state)))
