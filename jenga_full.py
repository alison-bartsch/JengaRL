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
		# Define action space - discrete action that can take on 51 values (id's of the jenga blocks)
		# the top three blocks should never be moved
		self.action_space = gym.spaces.Discrete(54) 
		# self.observation_space = np.ones(51)

		# the observation space is an adjacency matrix of dimension (54,54)
		self.observation_space = gym.spaces.MultiBinary((54,54)) 

		# Define the state - an adjacency matrix

		self.state = np.ones(51) 

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
		pb.removeBody(self.jengaObject[sampleID]) #delete selected block

		self.state[sampleID] = 0 #update state to describe remaining blocks
		# print("State Shape: ", self.state.shape)

		# place block on tower (call helper function to do so)

		num_blocks = 3+ np.sum(self.state)
		for _ in range(300): 
			pb.stepSimulation()

		# reward = 54 - num_blocks #increase reward for more blocks removed from tower
		reward = (54 - num_blocks)**2

		# due to the top 3 blocks never moved, we can use them to indicate the fall or not
		pos, ang = pb.getBasePositionAndOrientation(self.jengaObject[-1], self.physicsClient)
		if pos[2] >= 5:  # the accurate value should be 5.25, but we should take some viberation into consideration
			self.done = False
		else:
			reward = -100
			self.done = True
		outputs = [self.state, reward, self.done, dict()]

		print("\nAction: ", sampleID)
		print("Reward: ", reward)
		print("Num Removed: ", 54 - num_blocks)

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

		#block_height = 
		#block_length = 
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
		pos, ang = pb.getBasePositionAndOrientation(self.jengaObject[-1], self.physicsClient)

		# reset the state array
		self.state = np.ones(51) 

		return self.state

	# helper functions
	def _initialize_adjacency_matrix(self, num_blocks):
		state = np.zeros((num_blocks, num_blocks))



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

