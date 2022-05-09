import os
import gym
import pandas as pd
import numpy as np
# from jenga_discrete_voxelization import JengaEnv
import matplotlib.pyplot as plt
from jenga_full import JengaFullEnv
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback


# make directory to store stable baselines logs
log_dir = "./StableBaselines/A2C/"

# Parallel environments
env = JengaFullEnv()      # JengaFullEnv()    # JengaEnv()
env = Monitor(env, log_dir)

check_callback = CheckpointCallback(save_freq=500, save_path=log_dir, name_prefix='a2c_full_specialR')
model = A2C("MlpPolicy", env, verbose=1)


print("Beginning training...")
model.learn(total_timesteps=4000, callback=check_callback)
print("Ended training...")
model.save("a2c_jenga_full_specialR_4000_steps")



# # del model # remove to demonstrate saving and loading
# model = A2C.load("a2c_jenga_full_env_4000_steps")
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




        # env.render()