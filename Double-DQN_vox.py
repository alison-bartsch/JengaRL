import gym
import time
from matplotlib.cbook import flatten
import pybullet as p
from jenga_discrete_voxelization import JengaEnv

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque
import imageio

# The starter code follows the tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# we recommend you going through the tutorial before implement DQN algorithm


# define environment, please don't change 
env = JengaEnv()

# define transition tuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """
    define replay buffer class
    """
    def __init__(self, capacity):
        # raise NotImplementedError
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        # raise NotImplementedError
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # raise NotImplementedError
        return random.sample(self.memory, batch_size)

    def __len__(self):
        # raise NotImplementedError
        return len(self.memory)


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
        self.fc1 = nn.Linear(in_dim,200)
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,200)
        self.fc4 = nn.Linear(200,out_dim)
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

# hyper parameters you can play with
BATCH_SIZE = 16
GAMMA = 0.7
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 5
MEMORY_CAPACITY = 5000
PATH = './model.ckpt'
LR = 1e-5

n_actions = env.action_space.n
n_states = 9 * env.num_layer

# Here I used the Double-Q-learning
policy_net_a = DQN(n_states, n_actions)
policy_net_b = DQN(n_states, n_actions)

optimizer_a = optim.Adam(policy_net_a.parameters(), LR)
optimizer_b = optim.Adam(policy_net_b.parameters(), LR)
memory = ReplayMemory(MEMORY_CAPACITY)

steps_done = 0

def select_action(env,state):
    # given state, return the action with highest probability on the prediction of DQN model
    # you are recommended to also implement a soft-greedy here
    # raise NotImplementedError
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    # print(eps_threshold)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # if np.random.random() <= 0.5:
            #     Q_value_total = policy_net_a(state)

            # else:
            #     Q_value_total = policy_net_b(state)
            Q_value_total = policy_net_a(state) + policy_net_b(state)

            actions = torch.sort(Q_value_total,dim = 1)[1]
            for i in actions[0,:]:
                if i.item() in env.blocks_buffer:
                    action = i
                    return torch.tensor([[action.item()]])
    else:
        return torch.tensor([[np.random.choice(env.blocks_buffer)]], dtype=torch.long)


def optimize_model(policy_net,target_net,optimizer):
    # optimize the DQN model by sampling a batch from replay buffer
    # raise NotImplementedError
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    policy_net.train()
    target_net.eval()

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_actions = torch.argmax(policy_net(non_final_next_states),axis=1)
    next_state_values = torch.zeros(BATCH_SIZE,1)

    next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_state_actions.unsqueeze(1))

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.unsqueeze(1)

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# first burn in some experience
state = env.reset()
state = torch.from_numpy(state).float().view(1, -1)
# while len(memory) <= 200:
#     # Initialize the environment and state
    
#     # Select and perform an action
#     # print(state)
#     action = select_action(state)
#     new_state, reward, done, _ = env.step(action.item())
#     reward = torch.tensor([reward])

#     # # Observe new state
#     if not done:
#         next_state = torch.from_numpy(new_state).float().view(1, -1)
#         memory.push(state, action, next_state, reward)
#         state = next_state
#     else:
#         next_state = None
#         memory.push(state, action, next_state, reward)
#         state = env.reset()
#         state = torch.from_numpy(state).float().view(1, -1)

num_episodes = 150
episode_durations = []
def train(num_episodes):
    for i_episode in range(num_episodes):
        traj = []
        # Initialize the environment and state
        state = env.reset()
        state = torch.from_numpy(state).float().view(1, -1)
        for t in count():
            # Select and perform an action
            action = select_action(env,state)
            traj.append(action.item())
            new_state, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward])

            # # Observe new state
            if not done:
                next_state = torch.from_numpy(new_state).float().view(1, -1)
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            if np.random.random() <= 0.5:
                policy_net = policy_net_a
                target_net = policy_net_b
                optimizer = optimizer_a
            else:
                policy_net = policy_net_b
                target_net = policy_net_a
                optimizer = optimizer_b

            optimize_model(policy_net,target_net,optimizer)
            if done:
                print(traj)
                episode_durations.append(len(traj))
                print("Episode: {}, duration: {}".format(i_episode, len(traj)))
                break

        # if episode_durations[-1] == max(episode_durations):
        if i_episode % 5 == 0:
            # save the checkpoint
            torch.save({
                    'epoch': i_episode,
                    'model_state_dict_a': policy_net_a.state_dict(),
                    'model_state_dict_b': policy_net_b.state_dict(),
                    'optimizer_state_dict_a': optimizer_a.state_dict(),
                    'optimizer_state_dict_b': optimizer_b.state_dict(),
                    }, PATH)
            print("Save the best model with duration", episode_durations[-1])

train(num_episodes) 

# load the checkpoint
checkpoint = torch.load(PATH)
policy_net_a.load_state_dict(checkpoint['model_state_dict_a'])
policy_net_b.load_state_dict(checkpoint['model_state_dict_b'])
epoch = checkpoint['epoch']
policy_net_a.eval()
policy_net_b.eval()

# plot time duration
plt.figure()
plt.plot(np.arange(len(episode_durations)), episode_durations)
plt.show()

# visualize 
duration = []
frames = []
state = env.reset()
state = torch.from_numpy(state).float().view(1, -1)
# blocks_buffer = list(range(24))
for t in count():
    # env.render()
    # frames.append(env.render("rgb_array"))

    # Select and perform an action
    q_values = (policy_net_a(state) + policy_net_b(state))
    actions = torch.sort(q_values,dim = 1)[1]
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
        # episode_durations.append(t + 1)
        # print("Duration:", t+1)
        # duration.append(t+1)
        break
# print('The mean duration of all the 10 episodes during test is:',np.mean(duration))

# imageio.mimsave('./video.mp4', frames, 'MP4', fps=20)
# plt.show()
env.close()

