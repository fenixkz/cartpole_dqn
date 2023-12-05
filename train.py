import numpy as np
import gym
from buffer import ReplayBuffer
from dqn import DQN
import time
import matplotlib.pyplot as plt
import random
import os

if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    memory_capacity = 100

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    actions = range(action_size)
    policy_net = DQN(state_size, action_size)
    memory = ReplayBuffer(memory_capacity)
    n_episodes = 10
    n_steps = 250
    all_rewards = []
    epsilon = 0.99
    decay = 0.99
    model = policy_net.model
    sample_size = 2
    gamma = 0.9
    target_update_period = 10
    for e in range(n_episodes):
        total_reward = 0
        print("New episode #" + str(e))

        state = env.reset()[0].reshape(1,state_size)
        if epsilon > 0.05:
            epsilon *= decay
        for i in range(n_steps):
            if (random.random() < epsilon):
                action = np.random.choice(actions)
            else:
                q_values = model.predict(state)
                action = np.argmax(q_values)
            nextState, reward, done, x, _ = env.step(action)
            nextState = nextState.reshape(1, state_size)
            total_reward += reward
            memory.add(state, action, reward, done, nextState)
            state = nextState
            if memory.len() > sample_size:
                policy_net.replay(memory, gamma, sample_size)
            if done:
                break
        if e % target_update_period == 0:
            policy_net.target_update()
        all_rewards.append(total_reward)
        print("Total number of steps: %2d; Total reward: %2.2f; Epsilon: %3.2f" % (i + 1, total_reward, epsilon))
    t = np.arange(n_episodes)
    path = os.path.abspath(os.getcwd())
    name = str(path) + "/dqn_v1.h5"
    policy_net.model.save(name)
    plt.plot(t,all_rewards)
    plt.show()
