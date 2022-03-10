# Deep Q-network keras implementation to solve OpenAI gym Cartpole-v1 environment
### Requirements
In order to able to run the solution, few steps need to be verified/installed:
  - gym
    `pip3 install gym`
  - tensorflow
    `pip3 install tensorflow`
  - keras
    `pip3 install keras`
  - python 3.8.10
  - matplotlib
### Running
```
$ cd CartPole
$ python3 train.py
```

After that the agent starts learning. You can tune the following parameters:
  - Number of episodes (current 400)
  - sample_size (current 32)
  - gamma (current 0.9)
  - epsilon and epsilon' decay (0.99 both)

After the training completed, the plot of rewards over number of episodes is plotted. Also, the model is saved in the current directory under the name **dqn.h5**.
</br>
You can run `python3 evaluate.py` to see how long the model can balance the pole.

### Results
One of the possible results obtained:
![Rewards](https://github.com/fenixkz/cartpole_dqn/blob/main/Figure_1.png)

And finally, the maximum number of steps the model can balance is: 324
