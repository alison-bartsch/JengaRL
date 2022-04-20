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
		self.action_space = gym.spaces.Discrete(51) 

		# Define the state - cannot randomly initialize, because Jenga blocks are ordered
		# self.state=np.array(range(54))
		self.state = np.ones(51) 
		print("State: ", self.state)
		# physicsClient = pb.connect(pb.DIRECT)
		self.physicsClient = pb.connect(pb.GUI)
		pb.setTimeStep(1/60, self.physicsClient) # it's vital for stablity

		self.rendered_img = None
		self.done = None
		pb.setAdditionalSearchPath(pybullet_data.getDataPath())
		print(pybullet_data.getDataPath())

		self.reset()

	def render(self, mode='human'):
		if self.rendered_img is None:
			self.rendered_img = plt.imshow(np.zeros((1024, 1024, 4)))

		proj_matrix = pb.computeProjectionMatrixFOV(fov=90, aspect=1, nearVal=0.01, farVal=100)

		# want the camera view to be static
		cameraEyePosition = np.array([0.6, 0, 0.3])
		cameraTargetPosition = np.array([0, 0, 0.3])
		cameraUpVector = np.array([0, 0, 0.5])
		view_matrix = pb.computeViewMatrix(cameraEyePosition, cameraTargetPosition, cameraUpVector)

		# Display image
		frame = pb.getCameraImage(1024, 1024, view_matrix, proj_matrix)[2]
		frame = np.reshape(frame, (1024, 1024, 4))
		self.rendered_img.set_data(frame)
		plt.draw()
		plt.pause(0.1)
		# plt.pause(.00001)

	def step(self, sampleID):
		pb.removeBody(self.jengaObject[sampleID]) #delete selected block
		print(len(self.jengaObject))
		self.state[sampleID] = 0 #update state to describe remaining blocks
		# print("State Shape: ", self.state.shape)

		num_blocks = self.state.shape[0]
		for _ in range(300): 
			pb.stepSimulation()

		reward = 54 - num_blocks #increase reward for more blocks removed from tower

		# due to the top 3 blocks never moved, we can use them to indicate the fall or not
		pos, ang = pb.getBasePositionAndOrientation(self.jengaObject[-1], self.physicsClient)
		if pos[2] >= 5:  # the accurate value should be 5.25, but we should take some viberation into consideration
			self.done = False
		else:
			reward = -1000
			self.done = True
		outputs = [self.state, reward, self.done, dict()]
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
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[-(0.5),0,0+0.3*(layer+1)-0.15],baseOrientation=[0,0,0.7071,0.7071],useFixedBase= True,flags = pb.URDF_USE_SELF_COLLISION))
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[0,0,0+0.3*(layer+1)-0.15],baseOrientation=[0,0,0.7071,0.7071],useFixedBase= True,flags = pb.URDF_USE_SELF_COLLISION))
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[(0.5),0,0+0.3*(layer+1)-0.15],baseOrientation=[0,0,0.7071,0.7071],useFixedBase= True,flags = pb.URDF_USE_SELF_COLLISION))
			elif layer%2 ==1:
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[0,-(0.5),0+0.3*(layer+1)-0.15],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION))
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[0,0,0+0.3*(layer+1)-0.15],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION))
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[0,(0.5),0+0.3*(layer+1)-0.15],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION))
			else:
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[-(0.5),0,0+0.3*(layer+1)-0.15], baseOrientation=[0,0,0.7071,0.7071],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION))
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[0,0,0+0.3*(layer+1)-0.15], baseOrientation=[0,0,0.7071,0.7071],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION))
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[(0.5),0,0+0.3*(layer+1)-0.15], baseOrientation=[0,0,0.7071,0.7071],useFixedBase= fix_flag,flags = pb.URDF_USE_SELF_COLLISION))

		# self.rewardBoard = pb.loadURDF("jenga/jenga.urdf", basePosition=[0,0,.03*(18)+0.05], globalScaling=3)
		print("Created the Jenga Tower!")
		pos, ang = pb.getBasePositionAndOrientation(self.jengaObject[-1], self.physicsClient)
		print("Position: ", pos)
		return

# test code - see what is going on

# create a stable tower
# it's not a easy way!
env = JengaEnv()
done = False
for i in range(300):
	pb.stepSimulation()
	time.sleep(1./240.)

# random remove one jengas

print("Now start to remove the  jenga.")
while not done:
	action = env.action_space.sample()
	print("Action: ", action)
	state,rw,done,info = env.step(action)
	# print(state)
	print("Reward: ", rw)
# show what happened following
for i in range(300):
	pb.stepSimulation()
	time.sleep(1./240.)
# close the pybullet
pb.disconnect()




# import gym
# import numpy as np
# import pybullet as pb
# import matplotlib.pyplot as plt

# import pybullet_data

# # Discrete Case:
# class JengaEnv(gym.Env):
# 	metadata = {'render.modes': ['human']}

# 	def __init__(self):
# 		# Define action space - discrete action that can take on 54 values (id's of the jenga blocks)
# 		self.action_space = gym.spaces.Discrete(54)

# 		# Define the state - cannot randomly initialize, because Jenga blocks are ordered
# 		self.state = np.ones(3)	# 54

# 		physicsClient = pb.connect(pb.DIRECT)

# 		self.rendered_img = None

# 		pb.setAdditionalSearchPath(pybullet_data.getDataPath())
# 		print(pybullet_data.getDataPath())
# 		planeId = pb.loadURDF('plane.urdf')
# 		pb.setGravity(0, 0, -10, physicsClientId=physicsClient)
# 		jengaId = pb.loadURDF('jenga/jenga.urdf', basePosition=[0,-0.05,0+.025*(1)])
# 		print("Jenga ID: ", jengaId)
# 		block_measure = tuple(map(lambda i, j: i - j, pb.getAABB(jengaId)[1], pb.getAABB(jengaId)[0]))
# 		print("Block Measure: ", block_measure)

# 		#block_height = 
# 		#block_length = 
# 		self.jengaObject=[]
# 		for layer in range(1):	# 18
# 			if layer%2 ==1:
# 				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[0,-(0.05),0+.03*(layer+1)]))
# 				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[0,0,0+.03*(layer+1)]))
# 				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[0,(0.05),0+.03*(layer+1)]))
# 			else:
# 				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[-(0.05),0,0+.03*(layer+1)], baseOrientation=[0,0,0.7071,0.7071]))
# 				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[0,0,0+.03*(layer+1)], baseOrientation=[0,0,0.7071,0.7071]))
# 				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[(0.05),0,0+.03*(layer+1)], baseOrientation=[0,0,0.7071,0.7071]))

# 		print("Jenga Object Length", len(self.jengaObject))
# 		# self.rewardBoard = pb.loadURDF("jenga/jenga.urdf", basePosition=[0,0,.03*(18)+0.05], globalScaling=3)
# 		pb.stepSimulation()
# 		print("Created the Jenga Tower!")

# 	def render(self, mode='human', pause=0.00001):
# 		if self.rendered_img is None:
# 			self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

# 		proj_matrix = pb.computeProjectionMatrixFOV(fov=90, aspect=1, nearVal=0.01, farVal=100)

# 		# want the camera view to be static
# 		cameraEyePosition = np.array([0.6, 0, 0.3])
# 		cameraTargetPosition = np.array([0, 0, 0.3])
# 		cameraUpVector = np.array([0, 0, 0.5])
# 		view_matrix = pb.computeViewMatrix(cameraEyePosition, cameraTargetPosition, cameraUpVector)

# 		# Display image
# 		frame = pb.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
# 		frame = np.reshape(frame, (100, 100, 4))
# 		self.rendered_img.set_data(frame)
# 		plt.draw()
# 		plt.pause(pause)

# 	def step(self, sampleID):
# 		pb.removeBody(self.jengaObject[sampleID]) #delete selected block

# 		print(self.state.shape)

# 		self.state[sampleID] = 0

# 		print("state: ", self.state)

# 		# num_blocks = self.state.shape[0]
# 		num_blocks = np.sum(self.state)
# 		for _ in range(100): 
# 			pb.stepSimulation()
# 			self.render()


# 		reward = 54 - num_blocks #increase reward for more blocks removed from tower

# 		# print(pb.getBasePositionAndOrientation(self.rewardBoard)[0][2])
# 		# if pb.getBasePositionAndOrientation(self.rewardBoard)[0][2] > (0.03*17): #check if tower has fallen (if test board is still on top of the tower)
# 		# 	done = 0
# 		# 	print("Didn't Fall!")
# 		# else:
# 		# 	done = 1
# 		# 	reward = -1000
# 		# 	print("Tower Fell :(")
# 		# outputs = [self.state, reward, done]
# 		outputs = self.state, reward
# 		return outputs

# 	def reset(self):
# 		self.__init__()
# 		return


# # test code - see what is going on
# env = JengaEnv()
# env.render(pause=5)

# env.reset()

# for i in range(3):
# 	# remove the i+2 block
# 	print("Removing block...")
# 	state, reward = env.step(2-i)
# 	env.render(pause=3)

# 	# if done == 1:
# 	# 	assert False