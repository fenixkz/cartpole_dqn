from collections import deque
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen= size)

    def add(self, state, action, reward, done, nextState):
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
