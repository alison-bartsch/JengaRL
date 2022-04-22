import gym
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
		self.state = _initialize_adjacency_matrix(54)

		# Define helper information for continuing to stack blocks
		self.tower_layer = 17
		self.top_layer_ids = [51, 52, 53]
		self.top_layer_size = 3
		self.second_layer_ids = [48, 49, 50]

		self.physicsClient = pb.connect(pb.DIRECT)
		# self.physicsClient = pb.connect(pb.GUI)
		pb.setTimeStep(1/60, self.physicsClient) # it's vital for stablity
		self.rendered_img = None
		self.done = None
		pb.setAdditionalSearchPath(pybullet_data.getDataPath())
		print(pybullet_data.getDataPath())
		pb.setPhysicsEngineParameter(enableFileCaching=0)
		self.reset()

	def step(self, sampleID):
		# top_layer_ids, top_layer_size = _top_layer(self.jengaObject)

		# check if sampleID is in the top layer of the tower -- due to Jenga rules, you can't remove the top layer!
		if sampleID in self.top_layer_ids:
			reward = -50
			print("Tried to remove block from top of tower!")

		else:
			# remove block from tower
			pb.removeBody(self.jengaObject[sampleID]) #delete selected block
			# self.jengaObject[sampleID] = 0
			print("Jenga Object Length: ", len(self.jengaObject))
			# NOTE: may need to change the jengaObject to an array to maintain order as the blocks are stacked back onto the top

			# update the adjacency matrix after removing block from tower
			self.state = _remove_block_adjacency(54, sampleID, self.state) #update state to describe remaining blocks

			# place this block onto the top of the tower!
			if self.top_layer_size == 3:
				self.tower_layer+=1
				self.second_layer_ids = self.top_layer_ids
				self.top_layer_ids = [sampleID]
				# self.top_layer_size = 1
			else:
				self.top_layer_ids.append(sampleID)

			# intitialize spawning elements
			orientation = []
			position = []

			# if there are no blocks add block to center
			if self.top_layer_size == 3:
				position = [0,0,0+0.3*(layer+1)-0.15]

			# if there are 2 blocks, add block to one side
			elif self.top_layer_size == 2:
				posiiton = [0,-(0.5),0+0.3*(layer+1)-0.15]

			# if there is 1 block, add block to one side
			elif top_layer_size == 1:
				position = [0,(0.5),0+0.3*(layer+1)-0.15]

			if self.tower_layer % 2 == 1:
				# self.jengaObject[sampleID] = pb.loadURDF('jenga/jenga.urdf', basePosition=position, useFixedBase=False,flags = pb.URDF_USE_SELF_COLLISION)
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=position, useFixedBase=False,flags = pb.URDF_USE_SELF_COLLISION))
			else:
				orientation = [0,0,0.7071,0.7071]
				# self.jengaObject[sampleID] = pb.loadURDF('jenga/jenga.urdf', basePosition=position, baseOrientation=orientation,useFixedBase=False,flags = pb.URDF_USE_SELF_COLLISION)
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=position, baseOrientation=orientation,useFixedBase=False,flags = pb.URDF_USE_SELF_COLLISION))

			# update the size of the top layer
			self.top_layer_size+=1
			if self.top_layer_size > 3:
				self.top_layer_size = 1

			# update adjacency matrix
			self.state = _update_block_adjacency(sampleID, self.state, self.top_layer_ids, self.second_layer_ids)




			num_blocks = 3+ np.sum(self.state)
			reward = (54 - num_blocks)**2

		for _ in range(300): 
			pb.stepSimulation()

		# use the top most block as an indication if the tower is still standing
		idx = self.top_layer_ids[0]
		pos, ang = pb.getBasePositionAndOrientation(self.jengaObject[idx], self.physicsClient)
		if pos[2] > = math.floor(0+0.3*(self.tower_layer+1)-0.15):
			self.done = False
		else:
			reward = -100
			self.done = True
		outputs = [self.state, reward, self.done, dict()]

		print("\nAction: ", sampleID)
		print("Reward: ", reward)
		print("Num Removed: ", 54 - (3+ np.sum(self.state)))
		return outputs


	def reset(self):
		# self.__init__()
		pb.resetSimulation(self.physicsClient)
		pb.setGravity(0, 0, -10, physicsClientId=self.physicsClient)
		planeId = pb.loadURDF('plane.urdf')

		self.done = False
		# jengaId = pb.loadURDF('jenga/jenga.urdf', basePosition=[0,-0.05,0+.025*(1)])
		# block_measure = tuple(map(lambda i, j: i - j, pb.getAABB(jengaId)[1], pb.getAABB(jengaId)[0]))
		# print("Block Measure: ", block_measure)

		
		# self.jengaObject = np.zeros(54)
		# fix_flag = False
		# for layer in range(18):
		# 	if layer == 0:
		# 		self.jengaObject[layer] = pb.loadURDF('jenga/jenga.urdf', basePosition=[-(0.5),0,0+0.3*(layer+1)-0.15],baseOrientation=[0,0,0.7071,0.7071],useFixedBase= True,flags = pb.URDF_USE_SELF_COLLISION)
		# 		self.jengaObject[layer + 1] = pb.loadURDF('jenga/jenga.urdf', basePosition=[0,0,0+0.3*(layer+1)-0.15],baseOrientation=[0,0,0.7071,0.7071],useFixedBase= True,flags = pb.URDF_USE_SELF_COLLISION)
		# 		self.jengaObject[layer + 2] = pb.loadURDF('jenga/jenga.urdf', basePosition=[(0.5),0,0+0.3*(layer+1)-0.15],baseOrientation=[0,0,0.7071,0.7071],useFixedBase= True,flags = pb.URDF_USE_SELF_COLLISION)
		# 	elif layer%2 == 1:
		# 		self.jengaObject[layer] = pb.loadURDF('jenga/jenga.urdf', basePosition=[0,-(0.5),0+0.3*(layer+1)-0.15],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION)
		# 		self.jengaObject[layer + 1] = pb.loadURDF('jenga/jenga.urdf', basePosition=[0,0,0+0.3*(layer+1)-0.15],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION)
		# 		self.jengaObject[layer + 2] = pb.loadURDF('jenga/jenga.urdf', basePosition=[0,(0.5),0+0.3*(layer+1)-0.15],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION)
		# 	else:
		# 		self.jengaObject[layer] = pb.loadURDF('jenga/jenga.urdf', basePosition=[-(0.5),0,0+0.3*(layer+1)-0.15], baseOrientation=[0,0,0.7071,0.7071],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION)
		# 		self.jengaObject[layer + 1] = pb.loadURDF('jenga/jenga.urdf', basePosition=[0,0,0+0.3*(layer+1)-0.15], baseOrientation=[0,0,0.7071,0.7071],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION)
		# 		self.jengaObject[layer + 2] = pb.loadURDF('jenga/jenga.urdf', basePosition=[(0.5),0,0+0.3*(layer+1)-0.15], baseOrientation=[0,0,0.7071,0.7071],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION)


		self.jengaObject=[]
		fix_flag = False
		for layer in range(18): 
			if layer == 0:
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[-(0.5),0,0+0.3*(layer+1)-0.15],baseOrientation=[0,0,0.7071,0.7071],useFixedBase= True,flags = pb.URDF_USE_SELF_COLLISION)) # , globalScaling=10.0))
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[0,0,0+0.3*(layer+1)-0.15],baseOrientation=[0,0,0.7071,0.7071],useFixedBase= True,flags = pb.URDF_USE_SELF_COLLISION)) # , globalScaling=10.0))
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[(0.5),0,0+0.3*(layer+1)-0.15],baseOrientation=[0,0,0.7071,0.7071],useFixedBase= True,flags = pb.URDF_USE_SELF_COLLISION)) # , globalScaling=10.0))
			elif layer%2 ==1:
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[0,-(0.5),0+0.3*(layer+1)-0.15],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION)) # , globalScaling=10.0))
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[0,0,0+0.3*(layer+1)-0.15],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION)) # , globalScaling=10.0))
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[0,(0.5),0+0.3*(layer+1)-0.15],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION)) # , globalScaling=10.0))
			else:
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[-(0.5),0,0+0.3*(layer+1)-0.15], baseOrientation=[0,0,0.7071,0.7071],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION)) # , globalScaling=10.0))
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[0,0,0+0.3*(layer+1)-0.15], baseOrientation=[0,0,0.7071,0.7071],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION)) # , globalScaling=10.0))
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[(0.5),0,0+0.3*(layer+1)-0.15], baseOrientation=[0,0,0.7071,0.7071],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION)) # , globalScaling=10.0))

		# self.rewardBoard = pb.loadURDF("jenga/jenga.urdf", basePosition=[0,0,.03*(18)+0.05], globalScaling=3)
		print("Created the Jenga Tower!")
		# pos, ang = pb.getBasePositionAndOrientation(self.jengaObject[-1], self.physicsClient)

		# reset the state array
		self.state = _initialize_adjacency_matrix(54)

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

		print(state)
		return state

	def _remove_block_adjacency(self, n_blocks, sampleID, state):
		state[sampleID,:] = np.zeros(n_blocks)
		state[:, sampleID] = np.zeros(n_blocks)

		return state

	def _update_block_adjacency(self, sampleID, state, top_layer_ids, second_layer_ids):
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

	# def _top_layer(self, jengaObject):
	# 	# return the block id's in the top layer as well as the # of blocks in the top layer
	# 	# return top_layer_ids, top_layer_size

	# 	# get the 3 last elements of the JengaObject list
	# 	# pos, ang = getBasePositionAndOrientation(jengaObject[idx], self.physicsClient) ---- compare pos[2] for each of them

	# 	return top_layer_ids, top_layer_size






# # test code - see what is going on

# # create a stable tower
# # it's not a easy way!
# env = JengaEnv()
# done = False
# for i in range(300):
# 	print("Stepping the simulation")
# 	pb.stepSimulation()
# 	time.sleep(1./240.)

# # random remove one jengas

# print("Now start to remove the  jenga.")
# while not done:
# 	action = env.action_space.sample()
# 	print("Action: ", action)
# 	state,rw,done,info = env.step(action)
# 	# print(state)
# 	print("Reward: ", rw)
# # show what happened following
# for i in range(300):
# 	pb.stepSimulation()
# 	time.sleep(1./240.)
# # close the pybullet
# # pb.disconnect()

