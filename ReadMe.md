# Deep Q-Network and Double Deep Q-Network implementation with PyTorch
This project provides a PyTorch implementation of the Deep Q-Network ([DQN](https://arxiv.org/pdf/1312.5602.pdf)) and Double Deep Q-Network ([DDQN](https://arxiv.org/pdf/1509.06461.pdf)) reinforcement learning algorithms to solve the CartPole-v1 environment provided by OpenAI's Gym library.

### Prerequisites
The following dependencies are required to run this project:
  - python 3.8.10
  - gym
  - torch
  - matplotlib
  - gym[classic_control]

You can install these packages using pip

 - `pip3 install gym torch matplotlib`
 - `pip3 install gym[classic_control]`

### Launching
Navigate to the CartPole directory and run the *train.py* script:
```
$ cd CartPole
$ python3 train.py
```
This starts the training process for the agent. Upon completion of the training, the program will plot the history of rewards over the number of episodes. It will also save the model as **DQN.pth** in the *models* directory.

### Hyperparameters
You can tune the following hyperparameters to better train the model: 
  - Memory_size (default: 10000)
  - Sample_size (default: 64)
  - Number of episodes (default: 1000)
  - Number of steps (default: 500)
  - Gamma (default: 0.99)
  - Epsilon (default: 1)
  - Epsilon decay (default: 0.99)
  - Target network update period (default: 10)
  - Model to use (default: DDQN)

These hyperparameters are stored in *config/hyperparameters.py* file.

### Evaluation
</br>
To evaluate the performance of the trained model, you can run the `evaluate.py` script with several arguments:
  - --render to render the environment 
  - --untrained to use untrained network
  - --model with string argument to choose the model to test (DQN or DDQN) 

`python3 evaluate.py --render --model DDQN`

### Docker
Alternatively, you can run this repository inside the docker container without the need of installing the required packages. For that pull the image from the Docker Hub:

`docker pull fenixkz/cartpole_ddqn:torch`

And then run the bash script *start_docker.sh*

`chmod +x start_docker.sh`

`./start_docker.sh`

Note that if you want to use CUDA inside the docker image, you have to have **NVIDIA Container Toolkit**. The installation process can be found [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Also, this script runs the following command `xhost +local:root`. This command allows any local user to access your X server. It's a potential security risk, so be sure you understand the implications.

### Results
Here is an example of the potential results (DQN):
![Rewards](https://github.com/fenixkz/cartpole_dqn/blob/torch/figures/DQN_rewards.png)

In this instance, the maximum number of steps the model was able to balance the pole in evaluate mode was **235**.

Here is an example of training results for DDQN:
![Rewards](https://github.com/fenixkz/cartpole_dqn/blob/torch/figures/DDQN_rewards.png)

In this case, in evaluate mode the pole was balancing for hours