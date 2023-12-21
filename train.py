import numpy as np
import gym
from buffer import ReplayBuffer
from dqn import DQN, DDQN
import time
import matplotlib.pyplot as plt
import random
import os
import torch
from config import hyperparameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    env = gym.make("CartPole-v1")

    # Hyperparameters
    memory_capacity = hyperparameters.memory_size
    epsilon = hyperparameters.epsilon
    decay = hyperparameters.decay
    sample_size = hyperparameters.sample_size
    gamma = hyperparameters.gamma
    target_update_period = hyperparameters.target_update_period
    n_episodes = hyperparameters.n_episodes
    n_steps = hyperparameters.n_steps

    # Gym related parameters
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    actions = range(action_size)

    # DQN and Replay Buffer
    if hyperparameters.algorithm == "DDQN":
        policy_net = DDQN(state_size, action_size)
    else:
        policy_net = DQN(state_size, action_size)
    memory = ReplayBuffer(memory_capacity)

    all_rewards = []
    try:
        for e in range(n_episodes):
            total_reward = 0
            print("New episode #" + str(e))

            state = env.reset()[0].reshape(1,state_size)
            state = torch.from_numpy(state).float().to(device)
            if epsilon > 0.05:
                epsilon *= decay
            for i in range(n_steps):
                if (random.random() < epsilon):
                    action = np.random.choice(actions)
                else:
                    q_values = policy_net(state)
                    action = torch.argmax(q_values).item()
                nextState, reward, terminated, truncated, _ = env.step(action)
                done = truncated or terminated
                nextState = torch.from_numpy(nextState.reshape(1, state_size)).float().to(device)
                total_reward += reward
                memory.add(state, action, reward, done, nextState)
                state = nextState
                if memory.len() > sample_size:
                    policy_net.replay(memory, gamma, sample_size)
                if done:
                    break
            if e % target_update_period == 0:
                print("Updating the Target Network")
                policy_net.target_update()
            all_rewards.append(total_reward)
            print("Total number of steps: %2d; Total reward: %2.2f; Epsilon: %3.3f" % (i + 1, total_reward, epsilon))
    except KeyboardInterrupt:
        print("Training was interrupted by the user, saving the network ...")
    finally:
        # Save the model
        model_name = hyperparameters.algorithm + ".pth"
        model_dir = os.path.join(os.path.abspath(os.getcwd()), 'models')
        os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists
        name = os.path.join(model_dir, model_name)
        torch.save(policy_net.state_dict(), name)

        # Plot the reward history
        t = np.arange(len(all_rewards))
        window_size = 50  # Use the last 100 episodes for calculating the mean and std
        mean_rewards = np.array([np.mean(all_rewards[max(0, i - window_size + 1):(i+1)]) for i in range(n_episodes)])
        std_rewards = np.array([np.std(all_rewards[max(0, i - window_size + 1):(i+1)]) for i in range(n_episodes)])

        plt.figure()
        plt.plot(mean_rewards, color='blue')
        plt.fill_between(range(len(mean_rewards)), mean_rewards - std_rewards, mean_rewards + std_rewards, color='blue', alpha=0.2)
        plt.title('Reward history')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.savefig(hyperparameters.algorithm + "_rewards.png")
        plt.show()
