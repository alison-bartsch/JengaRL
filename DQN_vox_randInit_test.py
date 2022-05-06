import gym
import time
from matplotlib.cbook import flatten
import pybullet as p
from jenga_discrete_randomInit import JengaEnv

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque


# The starter code follows the tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# we recommend you going through the tutorial before implement DQN algorithm


# define environment, please don't change 
env = JengaEnv()

# define transition tuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



# hyper parameters you can play with

PATH = './model_12layer_best.ckpt'

class DQN(nn.Module):
    """
    build your DQN model:
    given the state, output the possiblity of actions
    """
    def __init__(self, in_dim, out_dim):
        """
        in_dim: dimension of states
        out_dim: dimension of actions
        """
        super(DQN, self).__init__()
        # build your model here
        self.fc1 = nn.Linear(in_dim,400)
        self.fc2 = nn.Linear(400,400)
        self.fc3 = nn.Linear(400,400)
        self.fc4 = nn.Linear(400,out_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # forward pass
        # raise NotImplementedError
        # print(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        # print(stop)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


n_actions = env.action_space.n
n_states = env.num_layer * 9

policy_net = DQN(n_states, n_actions)
target_net = DQN(n_states, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

episode_durations = []
# load the checkpoint
checkpoint = torch.load(PATH)
policy_net.load_state_dict(checkpoint['model_state_dict'])

policy_net.eval()

# plot time duration
plt.figure()
plt.plot(np.arange(len(episode_durations)), episode_durations)
plt.show()

# visualize 
duration = []
frames = []
for i in range(10):
    state = env.reset()
    state = torch.from_numpy(state).float().view(1, -1)

    for t in count():
        # env.render()
        # frames.append(env.render("rgb_array"))
    
        # Select and perform an action
        q_values = policy_net(state)
        actions = torch.sort(q_values,descending = True,dim = 1)[1]
    
        for i in actions[0,:]:
            if i.item() in env.blocks_buffer:
                action = i
                break
        print(action.item())
        # for 
        new_state, reward, done, _ = env.step(action.item())
        time.sleep(1/5)
        reward = torch.tensor([reward])
        # print(reward.item())
        # blocks_buffer.remove(action.item())
        # Observe new state
        if not done:
            next_state = torch.from_numpy(new_state).float().view(1, -1)
        else:
            next_state = None
    
        # Move to the next state
        state = next_state
    
        if done:
            episode_durations.append(t + 1)
            print("Duration:", t+1)
            duration.append(t+1)
            break
print('The mean duration of all the 10 episodes during test is:',np.mean(duration))

# imageio.mimsave('./video.mp4', frames, 'MP4', fps=20)
# plt.show()
env.close()

