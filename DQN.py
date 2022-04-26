import gym
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
import time
from jenga_discrete_voxelization import JengaEnv
# from jenga_discrete import JengaEnv

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
        self.fc1 = nn.Linear(in_dim,400)
        self.fc2 = nn.Linear(400,400)
        self.fc3 = nn.Linear(400,400)
        self.fc4 = nn.Linear(400,out_dim)
        self.relu = nn.LeakyReLU()

        #initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        # forward pass
        # raise NotImplementedError
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# hyper parameters you can play with
BATCH_SIZE = 64    #32 for 12 layer
GAMMA = 0.9        #0.9 for 12 layer
EPS_START = 0.9    #0.9 for 12 layer
EPS_END = 0.01        #0.01 for 12 layer
EPS_DECAY = 5000      # 200 for 12 layer
TARGET_UPDATE = 20    #10 for 12 layer
MEMORY_CAPACITY = 50000  
PATH = './model_12layer_best.ckpt'
LR = 8e-5        #1e-4 for 12 layer
num_episodes = 800   #500 for 12 layer

n_actions = env.action_space.n
n_states = env.num_layer * 9

policy_net = DQN(n_states, n_actions)
target_net = DQN(n_states, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), LR)
memory = ReplayMemory(MEMORY_CAPACITY)

steps_done = 0

def select_action(state):
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
            Q_value_total = policy_net(state)

            actions = torch.sort(Q_value_total,descending = True,dim = 1)[1]
            for i in actions[0,:]:
                if i.item() in env.blocks_buffer:
                    action = i
                    return torch.tensor([[action.item()]])
    else:
        return torch.tensor([[np.random.choice(env.blocks_buffer)]], dtype=torch.long)

def optimize_model():
    # optimize the DQN model by sampling a batch from replay buffer
    # raise NotImplementedError
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


episode_durations = []
# for i_episode in range(num_episodes):
#     # Initialize the environment and state
#     state = env.reset()
#     state = torch.from_numpy(state).float().view(1, -1)
#     for t in count():
#         # Select and perform an action
#         action = select_action(state)
#         new_state, reward, done, _ = env.step(action.item())
#         reward = torch.tensor([reward])

#         # # Observe new state
#         if not done:
#             next_state = torch.from_numpy(new_state).float().view(1, -1)
#         else:
#             next_state = None

    #     # Store the transition in memory
    #     memory.push(state, action, next_state, reward)
    #     # print(len(memory))

    #     # Move to the next state
    #     state = next_state

    #     # Perform one step of the optimization (on the target network)
    #     optimize_model()
    #     if done:
    #         episode_durations.append(t + 1)
    #         print("Episode: {}, duration: {}".format(i_episode, t+1))
    #         break
    
    # # Update the target network, copying all weights and biases in DQN
    # if i_episode % TARGET_UPDATE == 0:
    #     target_net.load_state_dict(policy_net.state_dict())

    # if episode_durations[-1] == max(episode_durations) or i_episode % 20 == 0:
    #     # save the checkpoint
    #     torch.save({
    #             'epoch': i_episode,
    #             'model_state_dict': policy_net.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             }, PATH)
    #     print("Save the best model with duration", episode_durations[-1])


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
state = env.reset()
state = torch.from_numpy(state).float().view(1, -1)
# blocks_buffer = list(range(24))
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
        # episode_durations.append(t + 1)
        # print("Duration:", t+1)
        # duration.append(t+1)
        break
# print('The mean duration of all the 10 episodes during test is:',np.mean(duration))

# imageio.mimsave('./video.mp4', frames, 'MP4', fps=20)
# plt.show()
env.close()
