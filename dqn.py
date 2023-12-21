import numpy as np
import gym
import os
import torch
import torch.nn as nn
import torch.optim as optim
import buffer
import random
import time

class DQN(nn.Module):
    def __init__(self, n_input, n_output):
        super(DQN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        # Define the model
        self.model = nn.Sequential(
            nn.Linear(n_input, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_output)
        ).to(self.device)
        # Define the target model
        self.target_model = nn.Sequential(
            nn.Linear(n_input, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_output)
        ).to(self.device)
        self.target_update()
        self.input_shape = n_input
        # Define loss function and optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def forward(self, x):
        return self.model(x)

    def replay(self, memory, gamma, sample_size):
        states, actions, rewards, dones, next_states = memory.sample(sample_size)
        # Preprocessing
        states = torch.from_numpy(np.asarray(states).reshape(-1, self.input_shape)).float().to(self.device)
        next_states = torch.from_numpy(np.asarray(next_states).reshape(-1, self.input_shape)).float().to(self.device)
        actions = torch.from_numpy(np.asarray(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.asarray(rewards)).float().to(self.device)
        dones = torch.from_numpy(np.asarray(dones)).float().to(self.device)

        # Calculate Q(s,a), shape (batch_size, action_size)
        q_values = self.model(states)
        # Choose only Q(s,a) of chosen actions, shape (batch_size, 1)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        # Calculate Q(s*,a), shape (batch_size, action_size)
        q_next = self.target_model(next_states)
        # Choose maximum of next Q(s*,a)
        q_next_max = q_next.max(1)[0]
        # Calculate target Y_t
        target = rewards + gamma * q_next_max * (1 - dones)

        # Update the network
        loss = self.loss_fn(q_values, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def target_update(self):
        self.target_model.load_state_dict(self.model.state_dict())


class DDQN(nn.Module):
    def __init__(self, n_input, n_output):
        super(DDQN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        # Define the model
        self.model = nn.Sequential(
            nn.Linear(n_input, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_output)
        ).to(self.device)
        # Define the target model
        self.target_model = nn.Sequential(
            nn.Linear(n_input, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_output)
        ).to(self.device)
        self.target_update()
        self.input_shape = n_input
        # Define loss function and optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def forward(self, x):
        return self.model(x)

    def replay(self, memory, gamma, sample_size):
        states, actions, rewards, dones, next_states = memory.sample(sample_size)
        # Preprocessing
        states = torch.from_numpy(np.asarray(states).reshape(-1, self.input_shape)).float().to(self.device)
        next_states = torch.from_numpy(np.asarray(next_states).reshape(-1, self.input_shape)).float().to(self.device)
        actions = torch.from_numpy(np.asarray(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.asarray(rewards)).float().to(self.device)
        dones = torch.from_numpy(np.asarray(dones)).float().to(self.device)

        # Calculate Q(s,a), shape (batch_size, action_size)
        q_values = self.model(states)
        # Choose only Q(s,a) of chosen actions, shape (batch_size, 1)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # DDQN Novelty: The main network chooses next best action
        next_best_actions = self.model(next_states).max(1)[1].unsqueeze(1)
        # Use target network to calculate the Q-values for that best actions
        q_next = self.target_model(next_states).gather(1, next_best_actions).squeeze(1)
        # Calculate target Y_t (DDQN)
        target = rewards + gamma * q_next * (1 - dones)

        # Update the network
        loss = self.loss_fn(q_values, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def target_update(self):
        self.target_model.load_state_dict(self.model.state_dict())
