# Size of a Replay Buffer
memory_size = 10000
# Batch size to sample from the Replay Buffer
sample_size = 64
# Number of episodes to train
n_episodes = 1000
# Number of maximum steps within one episode
n_steps = 500
# Epsilon (a.k.a. random factor in action choosing)
epsilon = 1
# Exponential decay of epsilon wrt to number of episodes
decay = 0.99
# Gamma (a.k.a short term vs long term reward)
gamma = 0.99
# Number of episodes to update the target network
target_update_period = 10
# What network to use (DQN, DDQN)
algorithm = "DDQN"
