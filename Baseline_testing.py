import gym
import time
from matplotlib.cbook import flatten
import pybullet as pb
from jenga_discrete_voxelization import JengaEnv

import pandas as pd
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from itertools import count


import imageio


SAVE_STR = 'random_baseline' 
# visualize 
duration = []
frames = []
count = 100
cumulative_r = []
episode_durations = []
env = JengaEnv()
for t in range(count):
    
    env.reset()
    done = False

    for i in range(300):
        pb.stepSimulation()
        time.sleep(1./240.)
    blocks_removed = -1
    r_total=0
    while not done:
        blocks_removed = blocks_removed + 1
        action = np.random.choice(env.blocks_buffer)
        
        state,rw,done,info = env.step(action)
        
        r_total+=rw
    # log reward
    cumulative_r.append(r_total)
    episode_durations.append(blocks_removed)
    print(blocks_removed)
    
    # show what happened following
    # save the data in a csv file for plotting later on
    for i in range(300):
        pb.stepSimulation()
        time.sleep(1./240.)
    # close the pybullet
pb.disconnect()
pd.DataFrame(cumulative_r).to_csv("./Data/" + SAVE_STR + 'baseline_reward.csv', header=None, index=None)
pd.DataFrame(episode_durations).to_csv("./Data/" + SAVE_STR + 'baseline_blocks_removed.csv', header=None, index=None)
