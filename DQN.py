import gym
import math
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from jenga_discrete import JengaEnv
# from jenga_full import JengaFullEnv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# The starter code follows the tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# we recommend you going through the tutorial before implement DQN algorithm


# define environment, please don't change 
env = JengaEnv()
# env = JengaFullEnv()

# define transition tuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """
    define replay buffer class
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
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

        # self.mlp = nn.Sequential(
        #     nn.Linear(in_dim, 125),
        #     nn.ReLU(),
        #     nn.Linear(125, 125),
        #     nn.ReLU(),
        #     nn.Linear(125, out_dim))

        self.fc1 = nn.Linear(in_dim, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 600)
        self.fc4 = nn.Linear(600, out_dim)
        self.relu = nn.LeakyReLU()

        # initialization
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.xavier_uniform(self.fc3.weight)
        nn.init.xavier_uniform(self.fc4.weight)

    def forward(self, x):
        # forward pass
        # return self.mlp(x)

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x






# hyper parameters you can play with
BATCH_SIZE = 64     # 128    
GAMMA = 0.9         # 0.95 
EPS_START = 0.9     # 0.99 
EPS_END = 0.01      # 0.012 
EPS_DECAY = 5000    # 2000 
TARGET_UPDATE = 20  # 10
MEMORY_CAPACITY = 50000     # 10000 
LEARNING_RATE = 8e-5
num_episodes = 4 # 800
SAVE_STR = 'disc_row_R'  # 'disc_std_R'     # 'disc_high_R'

n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

policy_net = DQN(n_states, n_actions)   # .to(device)
target_net = DQN(n_states, n_actions)   # .to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEMORY_CAPACITY)

steps_done = 0

def select_action(state):
    # given state, return the action with highest probability on the prediction of DQN model
    # you are recommended to also implement a soft-greedy here
    global steps_done
    sample = random.random()
    threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    steps_done+=1

    if sample > threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].unsqueeze(1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long) # , device=device)



def optimize_model():
    # optimize the DQN model by sampling a batch from replay buffer
    if len(memory) < BATCH_SIZE:
        return

    # sample transitions from the replay buffer
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # create a mask that can be used to eliminate all isntances of final states when indexing
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)

    # get the next states that are not final (where the simulation ended)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state = torch.cat(batch.state)
    action = torch.cat(batch.action)
    reward = torch.cat(batch.reward)

    # compute Q(s_t,a)
    Q_val = policy_net(state).gather(1, action)

    # compute V(s_{t+1}), where the expected next value will remain zero if the state is final
    V_next = torch.zeros(BATCH_SIZE)
    V_next[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # expected Q value
    Q_expected = (V_next * GAMMA) + reward

    # compute loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(Q_val, Q_expected.unsqueeze(1))

    # optimization
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1,1)
    optimizer.step()




num_blocks = [] # keeps track of the number of blocks removed
cumulative_r = []
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    state = torch.from_numpy(state).float().view(1, -1)

    r_total = 0

    for t in count():
        # Select and perform an action
        action = select_action(state)
        new_state, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward])

        # # Observe new state
        if not done:
            next_state = torch.from_numpy(new_state).float().view(1, -1)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        blocks_removed = 51 - np.sum(state.numpy())

        # Move to the next state
        state = next_state
        r_total+=reward

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            # print(state)
            num_blocks.append(blocks_removed)
            print(" ------------ Episode: {}, num blocks removed: {}".format(i_episode, blocks_removed))
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    cumulative_r.append(r_total.numpy())


# save the trained model
POLICY_PATH = './Models/' + SAVE_STR +'_policy.ckpt'
TARGET_PATH = './Models/' + SAVE_STR + '_target.ckpt'
torch.save(policy_net.state_dict(), POLICY_PATH)
torch.save(target_net.state_dict(), TARGET_PATH)

# save the data in a csv file for plotting later on
pd.DataFrame(cumulative_r).to_csv('./Data/' + SAVE_STR + '_reward.csv', header=None, index=None)
pd.DataFrame(num_blocks).to_csv('./Data/' + SAVE_STR + 'train_blocks_removed.csv', header=None, index=None)

# plot the reward over training
plt.figure()
plt.plot(np.arange(len(cumulative_r)), cumulative_r)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig('./Graphs/' + SAVE_STR + '_reward.png')

# calculate the moving average with a window of 50 episodes
window_size = 50
i = 0
moving_averages = []
while i < len(num_blocks) - window_size + 1:
    this_window = num_blocks[i : i + window_size]

    window_average = sum(this_window) / window_size
    moving_averages.append(window_average)
    i += 1

# plot time duration
plt.figure()
plt.plot(np.arange(len(num_blocks)), num_blocks)
plt.plot(np.arange(len(moving_averages)), moving_averages)
plt.xlabel("Episode")
plt.ylabel("Number of blocks removed from tower")
plt.legend(["After Each Episode", "Moving Average Over 50 Episodes"])
plt.savefig('./Graphs/' + SAVE_STR + '_blocks_removed.png')


vis_num_blocks = []


# visualize 
for i in range(10):
    # env.visualize()

    state = env.reset()
    state = torch.from_numpy(state).float().view(1, -1)
    for t in count():
        # env.render()

        # Select and perform an action
        action = select_action(state)
        new_state, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward])

        # Observe new state
        if not done:
            next_state = torch.from_numpy(new_state).float().view(1, -1)
        else:
            next_state = None

        blocks_removed = 51 - np.sum(state.numpy())

        # Move to the next state
        state = next_state

        if done:
            num_blocks.append(blocks_removed)
            vis_num_blocks.append(blocks_removed)
            # print("\nNumber of Blocks Removed:", blocks_removed)
            break

print("Mean # Blocks Removed Test Episodes: ", np.sum(vis_num_blocks) / len(vis_num_blocks))

# save data to csv
pd.DataFrame(vis_num_blocks).to_csv('./Data/' + SAVE_STR + 'test_blocks_removed.csv', header=None, index=None)

env.close()


