import gym
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt

import pybullet_data

# Discrete Case:
class JengaEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		# Define action space - discrete action that can take on 54 values (id's of the jenga blocks)
		self.action_space = gym.spaces.Discrete(54)

		# Define the state - cannot randomly initialize, because Jenga blocks are ordered
		self.state=np.array(range(54)) 
		print("State: ", self.state)
		physicsClient = pb.connect(pb.DIRECT)

		self.rendered_img = None

		pb.setAdditionalSearchPath(pybullet_data.getDataPath())
		print(pybullet_data.getDataPath())
		planeId = pb.loadURDF('plane.urdf')
		pb.setGravity(0, 0, -10, physicsClientId=physicsClient)
		jengaId = pb.loadURDF('jenga/jenga.urdf', basePosition=[0,-0.05,0+.025*(1)])
		block_measure = tuple(map(lambda i, j: i - j, pb.getAABB(jengaId)[1], pb.getAABB(jengaId)[0]))
		print("Block Measure: ", block_measure)

		#block_height = 
		#block_length = 
		self.jengaObject=[]
		for layer in range(18):
			if layer%2 ==1:
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[0,-(0.05),0+.03*(layer+1)]))
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[0,0,0+.03*(layer+1)]))
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[0,(0.05),0+.03*(layer+1)]))
			else:
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[-(0.05),0,0+.03*(layer+1)], baseOrientation=[0,0,0.7071,0.7071]))
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[0,0,0+.03*(layer+1)], baseOrientation=[0,0,0.7071,0.7071]))
				self.jengaObject.append(pb.loadURDF('jenga/jenga.urdf', basePosition=[(0.05),0,0+.03*(layer+1)], baseOrientation=[0,0,0.7071,0.7071]))

		self.rewardBoard = pb.loadURDF("jenga/jenga.urdf", basePosition=[0,0,.03*(18)+0.05], globalScaling=3)
		pb.stepSimulation()
		print("Created the Jenga Tower!")

	def render(self, mode='human'):
		if self.rendered_img is None:
			self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

		proj_matrix = pb.computeProjectionMatrixFOV(fov=90, aspect=1, nearVal=0.01, farVal=100)

		# want the camera view to be static
		cameraEyePosition = np.array([0.6, 0, 0.3])
		cameraTargetPosition = np.array([0, 0, 0.3])
		cameraUpVector = np.array([0, 0, 0.5])
		view_matrix = pb.computeViewMatrix(cameraEyePosition, cameraTargetPosition, cameraUpVector)

		# Display image
		frame = pb.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
		frame = np.reshape(frame, (100, 100, 4))
		self.rendered_img.set_data(frame)
		plt.draw()
		plt.pause(3)
		# plt.pause(.00001)

	def step(self, sampleID):
		pb.removeBody(self.jengaObject[sampleID]) #delete selected block

		self.state = self.state[self.state != sampleID] #update state to describe remaining blocks
		# print("State Shape: ", self.state.shape)

		num_blocks = self.state.shape[0]
		for _ in range(100): 
			pb.stepSimulation()

		reward = 54 - num_blocks #increase reward for more blocks removed from tower

		print(pb.getBasePositionAndOrientation(self.rewardBoard)[0][2])
		if pb.getBasePositionAndOrientation(self.rewardBoard)[0][2] > (0.03*17): #check if tower has fallen (if test board is still on top of the tower)
			done = 0
			print("Didn't Fall!")
		else:
			done = 1
			reward = -1000
			print("Tower Fell :(")
		outputs = [self.state, reward, done]
		return outputs

	def reset(self):
		self.__init__()
		return


# test code - see what is going on
env = JengaEnv()
env.render()

env.reset()

for i in range(10):
	# remove the i+2 block
	print("Removing block...")
	state, reward, done = env.step(i+2)
	env.render()

	if done == 1:
		assert False