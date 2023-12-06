# Double Deep Q-network keras implementation to solve OpenAI gym Cartpole-v1 environment
This project provides a Keras implementation of the Double Deep Q-Network (DDQN) algorithm to solve the CartPole-v1 environment provided by OpenAI's Gym library.
### Prerequisites
The following dependencies are required to run this project:
  - python 3.8.10
  - gym
  - tensorflow
  - keras
  - matplotlib
  - gym[classic_control]

You can install these packages using pip

 - `pip3 install gym tensorflow keras matplotlib`
 - `pip3 install gym[classic_control]`

### Usage
Navigate to the CartPole directory and run the *train.py* script:
```
$ cd CartPole
$ python3 train.py
```
This starts the training process for the agent. You can modify the following parameters:
  - Number of episodes (default: 400)
  - Sample_size (default: 32)
  - Gamma (default: 0.9)
  - Epsilon and epsilon' decay (default: 0.99 for both)

Upon completion of the training, the program will plot the rewards over the number of episodes. It will also save the model as **dqn.h5** in the *models* directory.
</br>
To evaluate the performance of the trained model, you can run the *evaluate.py* script with several arguments:
  - --render to render the environment 
  - -- untrained to use untrained network

`python3 evaluate.py --render --untrained`

</br>
You can also
### Results
Here is an example of the potential results:
![Rewards](https://github.com/fenixkz/cartpole_dqn/blob/main/Figure_1.png)

In this instance, the maximum number of steps the model was able to balance the pole was **324**.
