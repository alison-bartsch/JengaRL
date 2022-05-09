import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


random = pd.read_csv("./Data/voxelized_random.csv")
print(random)
random=random.to_numpy()
plt.figure()
plt.plot(np.arange(len(random)), random)
plt.xlabel("Episode")
plt.ylabel("Blocks Removed")
plt.show()