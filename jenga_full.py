import gym
import math
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt
import time
import pybullet_data

# Discrete Case:
class JengaEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		# Define action space - discrete action that can take on 54 values (id's of the jenga blocks)
		self.action_space = gym.spaces.Discrete(54) 

		# the observation space is an adjacency matrix of dimension (54,54)
		self.observation_space = gym.spaces.MultiBinary((54,54)) 

		# Define the state - an adjacency matrix
		self.state = self._initialize_adjacency_matrix(54)

		# Define helper information for continuing to stack blocks
		self.tower_layer = 17
		self.top_layer_ids = [51, 52, 53]
		self.top_layer_size = 3
		self.second_layer_ids = [48, 49, 50]

		# number of blocks removed counter
		self.num_removed = 0

		# self.physicsClient = pb.connect(pb.DIRECT)
		self.physicsClient = pb.connect(pb.GUI)

		pb.setTimeStep(1/60, self.physicsClient) # it's vital for stablity
		self.rendered_img = None
		self.done = None
		pb.setAdditionalSearchPath(pybullet_data.getDataPath())
		print(pybullet_data.getDataPath())
		pb.setPhysicsEngineParameter(enableFileCaching=0)
		self.reset()

	def step(self, sampleID):

		# check if sampleID is in the top layer of the tower -- due to Jenga rules, you can't remove the top layer!
		if sampleID in self.top_layer_ids:
			reward = -50
			print("Tried to remove block from top of tower!")

		else:
			# remove block from tower
			print("\nSampel ID: ", sampleID)
			print("Int of Jenga Object: ", int(self.jengaObject[sampleID]))
			print("Jenga Object: ", self.jengaObject[sampleID])
			pb.removeBody(int(self.jengaObject[sampleID])) #delete selected block
			self.jengaObject[sampleID] = 0

			# update the adjacency matrix after removing block from tower
			self.state = self._remove_block_adjacency(54, sampleID, self.state) #update state to describe remaining blocks

			# place this block onto the top of the tower!
			if self.top_layer_size == 3:
				self.tower_layer+=1
				self.second_layer_ids = self.top_layer_ids
				self.top_layer_ids = [sampleID]
			else:
				self.top_layer_ids.append(sampleID)

			print("Tower Layer: ", self.tower_layer)
			print("Top Layer ID's: ", self.top_layer_ids)
			print("Second Layer ID's: ", self.second_layer_ids)

			# intitialize spawning elements
			orientation = []
			position = []

			# if there are no blocks add block to center
			if self.top_layer_size == 3:
				print("3 blocks here!")
				position = [0,0,0+0.3*(self.tower_layer+1)-0.2]

			# if there are 2 blocks, add block to one side
			elif self.top_layer_size == 2:
				print("2 blocks here!")
				position = [-(0.5),0,0+0.3*(self.tower_layer+1)-0.2]

			# if there is 1 block, add block to one side
			elif self.top_layer_size == 1:
				print("1 block here!")
				position = [(0.5),0,0+0.3*(self.tower_layer+1)-0.2]

			if self.tower_layer % 2 == 1:
				print("odd tower layer")
				self.jengaObject[sampleID] = pb.loadURDF('jenga/jenga.urdf', basePosition=position, useFixedBase=False,flags = pb.URDF_USE_SELF_COLLISION)
			else:
				print("even tower layer")
				orientation = [0,0,0.7071,0.7071]
				self.jengaObject[sampleID] = pb.loadURDF('jenga/jenga.urdf', basePosition=position, baseOrientation=orientation,useFixedBase=False,flags = pb.URDF_USE_SELF_COLLISION)
		
			# update the size of the top layer
			self.top_layer_size+=1
			if self.top_layer_size > 3:
				self.top_layer_size = 1

			# update adjacency matrix
			self.state = self._update_block_adjacency(sampleID, self.state, self.top_layer_ids, self.second_layer_ids, self.top_layer_size)

			self.num_removed+=1
			reward = (self.num_removed)**2

		for _ in range(300): 
			pb.stepSimulation()

		# use the top most block as an indication if the tower is still standing
		idx = self.top_layer_ids[0]
		pos, ang = pb.getBasePositionAndOrientation(int(self.jengaObject[idx]), self.physicsClient)
		if pos[2] >= math.floor(0+0.3*(self.tower_layer+1)-0.15):
			self.done = False
		else:
			reward = -100
			self.done = True
		outputs = [self.state, reward, self.done, dict()]

		print("Action: ", sampleID)
		print("Reward: ", reward)
		print("Num Removed: ", self.num_removed)
		return outputs


	def reset(self):
		# self.__init__()
		pb.resetSimulation(self.physicsClient)
		pb.setGravity(0, 0, -10, physicsClientId=self.physicsClient)
		planeId = pb.loadURDF('plane.urdf')

		self.done = False

		self.jengaObject = np.zeros(54)
		fix_flag = False
		for layer in range(18):
			if layer == 0:
				self.jengaObject[layer] = pb.loadURDF('jenga/jenga.urdf', basePosition=[-(0.5),0,0+0.3*(layer+1)-0.15],baseOrientation=[0,0,0.7071,0.7071],useFixedBase= True,flags = pb.URDF_USE_SELF_COLLISION)
				self.jengaObject[layer + 1] = pb.loadURDF('jenga/jenga.urdf', basePosition=[0,0,0+0.3*(layer+1)-0.15],baseOrientation=[0,0,0.7071,0.7071],useFixedBase= True,flags = pb.URDF_USE_SELF_COLLISION)
				self.jengaObject[layer + 2] = pb.loadURDF('jenga/jenga.urdf', basePosition=[(0.5),0,0+0.3*(layer+1)-0.15],baseOrientation=[0,0,0.7071,0.7071],useFixedBase= True,flags = pb.URDF_USE_SELF_COLLISION)
			elif layer%2 == 1:
				self.jengaObject[layer*3] = pb.loadURDF('jenga/jenga.urdf', basePosition=[0,-(0.5),0+0.3*(layer+1)-0.15],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION)
				self.jengaObject[layer*3 + 1] = pb.loadURDF('jenga/jenga.urdf', basePosition=[0,0,0+0.3*(layer+1)-0.15],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION)
				self.jengaObject[layer*3 + 2] = pb.loadURDF('jenga/jenga.urdf', basePosition=[0,(0.5),0+0.3*(layer+1)-0.15],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION)
			else:
				self.jengaObject[layer*3] = pb.loadURDF('jenga/jenga.urdf', basePosition=[-(0.5),0,0+0.3*(layer+1)-0.15], baseOrientation=[0,0,0.7071,0.7071],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION)
				self.jengaObject[layer*3 + 1] = pb.loadURDF('jenga/jenga.urdf', basePosition=[0,0,0+0.3*(layer+1)-0.15], baseOrientation=[0,0,0.7071,0.7071],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION)
				self.jengaObject[layer*3 + 2] = pb.loadURDF('jenga/jenga.urdf', basePosition=[(0.5),0,0+0.3*(layer+1)-0.15], baseOrientation=[0,0,0.7071,0.7071],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION)

		print("Jenga Object: ", self.jengaObject)
		print("Created the Jenga Tower!")

		# reset the state array
		self.state = self._initialize_adjacency_matrix(54)

		# reset the helper information
		self.tower_layer = 17
		self.top_layer_ids = [51, 52, 53]
		self.top_layer_size = 3
		self.second_layer_ids = [48, 49, 50]

		return self.state

	# helper functions
	def _initialize_adjacency_matrix(self, num_blocks):
		state = np.zeros((num_blocks, num_blocks))

		# iterate over every 3 numbers in range(54)
		for i in np.arange(num_blocks)[::3]:
			
			# check for bottom layer
			if i == 0:
				state[i+1, i] = 1
				state[i+3, i] = 1
				state[i+4, i] = 1
				state[i+5, i] = 1

				state[i, i+1] = 1
				state[i+2, i+1] = 1
				state[i+3, i+1] = 1
				state[i+4, i+1] = 1
				state[i+5, i+1] = 1

				state[i+1, i+2] = 1
				state[i+3, i+2] = 1
				state[i+4, i+2] = 1
				state[i+5, i+2] = 1

			# check for top layer
			elif i == num_blocks - 3:
				state[i-3, i] = 1
				state[i-2, i] = 1
				state[i-1, i] = 1
				state[i+1, i] = 1

				state[i-3, i+1] = 1
				state[i-2, i+1] = 1
				state[i-1, i+1] = 1
				state[i, i+1] = 1
				state[i+2, i+1] = 1

				state[i-3, i+2] = 1
				state[i-2, i+2] = 1
				state[i-1, i+2] = 1
				state[i+1, i+2] = 1

			else:
				state[i-3, i] = 1
				state[i-2, i] = 1
				state[i-1, i] = 1
				state[i+1, i] = 1
				state[i+3, i] = 1
				state[i+4, i] = 1
				state[i+5, i] = 1

				state[i-3, i+1] = 1
				state[i-2, i+1] = 1
				state[i-1, i+1] = 1
				state[i, i+1] = 1
				state[i+2, i+1] = 1
				state[i+3, i+1] = 1
				state[i+4, i+1] = 1
				state[i+5, i+1] = 1

				state[i-3, i+2] = 1
				state[i-2, i+2] = 1
				state[i-1, i+2] = 1
				state[i+1, i+2] = 1
				state[i+3, i+2] = 1
				state[i+4, i+2] = 1
				state[i+5, i+2] = 1

		return state

	def _remove_block_adjacency(self, n_blocks, sampleID, state):
		state[sampleID,:] = np.zeros(n_blocks)
		state[:, sampleID] = np.zeros(n_blocks)
		return state

	def _update_block_adjacency(self, sampleID, state, top_layer_ids, second_layer_ids, top_layer_size):
		if top_layer_size == 1:
			# this means the only block on the top layer is the sample ID, so it only needs to be connected to second_layer_ids
			state[sampleID, second_layer_ids[0]] = 1
			state[sampleID, second_layer_ids[1]] = 1
			state[sampleID, second_layer_ids[2]] = 1
			state[second_layer_ids[0], sampleID] = 1
			state[second_layer_ids[1], sampleID] = 1
			state[second_layer_ids[2], sampleID] = 1

		else:
			# this means the sampleID is on the left or right of the center block 
			# because it is a list that is appended, the center block should be top_layer_ids[0]

			# the relationships between sampleID and the blocks underneath
			state[sampleID, second_layer_ids[0]] = 1
			state[sampleID, second_layer_ids[1]] = 1
			state[sampleID, second_layer_ids[2]] = 1
			state[second_layer_ids[0], sampleID] = 1
			state[second_layer_ids[1], sampleID] = 1
			state[second_layer_ids[2], sampleID] = 1

			# the relationship between sampleID and the block next to it
			state[sampleID, top_layer_ids[0]] = 1
			state[top_layer_ids[0], sampleID] = 1

		return state



# test code - see what is going on

# create a stable tower
# it's not a easy way!
env = JengaEnv()
done = False
for i in range(300):
	# print("Stepping the simulation")
	pb.stepSimulation()
	time.sleep(1./240.)

# random remove one jengas

print("Now start to remove the  jenga.")
while not done:
	action = env.action_space.sample()
	state,rw,done,info = env.step(action)
	# print(state)
	# print("Reward: ", rw)
# show what happened following
for i in range(300):
	pb.stepSimulation()
	time.sleep(1./240.)
# close the pybullet
# # pb.disconnect()

