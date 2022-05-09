import os
import gym
import pandas as pd
import numpy as np
# from jenga_discrete_voxelization import JengaEnv
import matplotlib.pyplot as plt
from jenga_full import JengaFullEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
# from stable_baselines3.common.results_plotter import load_results, ts2xy



# make directory to store stable baselines logs
log_dir = "./StableBaselines/PPO/"

# Parallel environments
env = JengaFullEnv()    # JengaFullEnv()    # JengaEnv()
env = Monitor(env, log_dir)

check_callback = CheckpointCallback(save_freq=200, save_path=log_dir, name_prefix='ppo_full_specialR')
model = PPO("MlpPolicy", env, verbose=1)


print("Beginning training...")
model.learn(total_timesteps=4000, callback=check_callback)
print("Ended training...")
model.save("ppo_jenga_full_specialR_4000_steps")




# model = PPO.load("ppo_jenga_vox_4000_steps")

# blocks_removed = []
# for i in range(75):

#     obs = env.reset()
#     n_blocks = 0
#     while True:
#         action, _states = model.predict(obs)
#         obs, rewards, dones, info = env.step(action)

#         if rewards > 0:
#             n_blocks+=1

#         if dones:
#             break

#     blocks_removed.append(n_blocks)

# print("Average blocks removed", np.mean(blocks_removed))
# print("Max blocks removed", np.amax(blocks_removed))








# del model # remove to demonstrate saving and loading
# model = PPO.load("ppo_jenga_vox")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     # env.render()




# # Plot the reward/number of blocks removed
# SAVE_STR = 'PPO_full_jenga'
# csv_loc = log_dir + 'monitor.csv'

# # import data from monitor.csv
# monitor = pd.read_csv(csv_loc, sep=',',header=None).to_numpy()
# print(monitor)
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




