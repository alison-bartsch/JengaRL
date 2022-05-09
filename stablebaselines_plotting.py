import os
import gym
import pandas as pd
import numpy as np
# from jenga_discrete_voxelization import JengaEnv
import matplotlib.pyplot as plt
# from jenga_full import JengaFullEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

print("Hello!")

# random = pd.read_csv("./Data/voxelized_random.csv")
# plt.figure()
# plt.plot(np.arange(len(random)), random)
# plt.xlabel("Episode")
# plt.ylabel("Blocks Removed")
# plt.show()



# log_dir = "./StableBaselines/PPO/"			# "./StableBaselines/PPO/"		# "./StableBaselines/A2C/"

# # Plot the reward/number of blocks removed
# SAVE_STR = 'PPO_full_jenga_4000'		# 'PPO_full_jenga'. A2C_full_jenga'
# csv_loc = log_dir + 'monitor_full.csv'

# # import data from monitor.csv
# monitor = pd.read_csv(csv_loc).values
# print(monitor.shape)
# n_blocks_removed = monitor[:,1] - 1
# cumulative_r = monitor[:,0]
# print(n_blocks_removed)
# print(cumulative_r)

# # plot blocks removed
# plt.figure()
# plt.plot(np.arange(len(n_blocks_removed)), n_blocks_removed)
# plt.xlabel("Episode")
# plt.ylabel("Num Blocks from Tower")
# plt.savefig('./Graphs/' + SAVE_STR + '_blocks_removed.png')

# # plot reward
# plt.figure()
# plt.plot(np.arange(len(cumulative_r)), cumulative_r)
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.savefig('./Graphs/' + SAVE_STR + '_reward.png')



# CODE TO turn the montior files to 
log_dir = "./StableBaselines/A2C/"
SAVE_STR = 'A2C_vox_specialR_'	
csv_loc = log_dir + 'monitor_vox_special.csv'


monitor = pd.read_csv(csv_loc).values
n_blocks_removed = monitor[:,1] - 1
cumulative_r = monitor[:,0]

pd.DataFrame(cumulative_r).to_csv('./TrainedCSVInfo/VoxJenga/' + SAVE_STR + '_reward.csv', header=None, index=None)
pd.DataFrame(n_blocks_removed).to_csv('./TrainedCSVInfo/VoxJenga/' + SAVE_STR + '_blocks_removed.csv', header=None, index=None)



