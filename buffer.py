from collections import deque, namedtuple
import numpy as np
import random
import torch

class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen= size)

    def add(self, state, action, reward, done, nextState):
        state = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
        nextState = nextState.cpu().numpy() if isinstance(nextState, torch.Tensor) else nextState
        exp = (state, action, reward, done, nextState)
        self.buffer.append(exp)

    def sample(self, sample_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        nextState_batch = []

        batch = random.sample(self.buffer, sample_size)
        for state, action, reward, done, nextState in batch:
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            done_batch.append(done)
            nextState_batch.append(nextState)
        return state_batch, action_batch, reward_batch, done_batch, nextState_batch

    def len(self):
        return len(self.buffer)

    def print_memory(self):
        print(self.buffer)

class PrioritizedReplayBuffer:
    def __init__(self, size, alpha=0.6, beta_start=0.4, beta_increment=0.001):
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = beta_increment
        self.buffer = deque(maxlen=size)
        self.priorities = deque(maxlen=size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "done", "next_state"])

    def add(self, state, action, reward, done, next_state):
        state = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
        next_state = next_state.cpu().numpy() if isinstance(next_state, torch.Tensor) else next_state
        exp = self.experience(state, action, reward, done, next_state)
        
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append(exp)
        self.priorities.append(max_priority)

    def sample(self, sample_size):
        probabilities = np.array(self.priorities) ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), sample_size, p=probabilities)
        experiences = [self.buffer[index] for index in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        states = [e.state for e in experiences]
        actions = [e.action for e in experiences]
        rewards = [e.reward for e in experiences]
        dones = [e.done for e in experiences]
        next_states = [e.next_state for e in experiences]
        
        return states, actions, rewards, dones, next_states, indices, weights

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            self.priorities[index] = priority

    def __len__(self):
        return len(self.buffer)

    def print_memory(self):
        print(self.buffer)