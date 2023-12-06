import numpy as np
import gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
import buffer
import random
import time


class DQN:
    def __init__(self, n_input, n_output):
        self.input_shape = n_input
        self.output_shape = n_output
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_update()

    def create_model(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=self.input_shape))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.output_shape, activation='linear'))
        model.compile(loss='mse', optimizer = Adam(0.001))
        return model

    def replay(self, memory, gamma, sample_size):
        states, actions, rewards, dones, nextStates = memory.sample(sample_size)

        states = np.asarray(states).reshape(-1, self.input_shape)
        nextStates = np.asarray(nextStates).reshape(-1, self.input_shape)
        q_values = self.model.predict(states) # (sample_size, output_shape)
        q_nexts = self.target_model.predict(nextStates)
        for i in range(sample_size):
            if (dones[i]):
                q_values[i][actions[i]] = rewards[i]
            else:
                q_values[i][actions[i]] = rewards[i] + gamma * np.max(q_nexts[i])
        self.model.fit(states, q_values, epochs = sample_size, verbose = 0)

    def target_update(self):
        self.target_model.set_weights(self.model.get_weights())
