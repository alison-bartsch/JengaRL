import gym
import numpy as np
import pybullet as pb
import matplotlib.pyplot as plt
import time
import pybullet_data
import random

# Discrete Case:
class JengaEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		# Define action space - discrete action that can take on 51 values (id's of the jenga blocks)
		# the top three blocks should never be moved

		self.num_layer = 12 # here is the only thing that you need to change
		self.done = False
		self.num_blocks = 3 * self.num_layer - 3 #top 3 blocks are elimited

		self.action_space = gym.spaces.Discrete(self.num_blocks) #total:51
		self.observation_space = gym.spaces.box.Box(
			low=np.zeros((3,3,self.num_layer), dtype=np.float32),
            high=np.ones((3,3,self.num_layer), dtype=np.float32))

		# Define the state - cannot randomly initialize, because Jenga blocks are ordered
		# self.state=np.array(range(54))
		
		# print("State: ", self.state)
		# self.physicsClient = pb.connect(pb.DIRECT)
		self.physicsClient = pb.connect(pb.GUI)

		pb.setTimeStep(1/60, self.physicsClient) # it's vital for stablity

		self.rendered_img = None
		self.done = None
		self.num_blocks = None
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
		#delete selected block
		pb.removeBody(self.jengaObject[sampleID]) 

		# get the position of the sample
		layer = int(sampleID / 3)
		pos = sampleID - int(sampleID / 3) * 3

		# update the state
		if layer % 2 == 0:
			self.state[:,pos,layer] = 0
		else:
			self.state[pos,:,layer] = 0

		# print("State Shape: ", self.state.shape)
		self.blocks_buffer.remove(sampleID)

		self.num_blocks -= 1

		for _ in range(90): 
			pb.stepSimulation()

		reward = (3 * self.num_layer - 3 - self.num_blocks)**2 + sampleID #increase reward for more blocks removed from tower

		# due to the top 3 blocks never moved, we can use them to indicate the fall or not
		pos, ang = pb.getBasePositionAndOrientation(self.jengaObject[-1], self.physicsClient)
		if pos[2] >= 0.3 * self.num_layer - 0.5:  # the accurate value should be 5.25, but we should take some viberation into consideration
			self.done = False
		else:
			reward = -100
			self.done = True

		outputs = [self.state, reward, self.done, dict()]
		return outputs

	def reset(self):
		# self.__init__()
		pb.resetSimulation(self.physicsClient)
		pb.setGravity(0, 0, -10, physicsClientId=self.physicsClient)
		planeId = pb.loadURDF('plane.urdf')

		self.state = np.ones((3,3, self.num_layer)) 
		self.done = False
		self.num_blocks = 3 * self.num_layer - 3
		self.blocks_buffer = list(range(self.num_blocks))
		# jengaId = pb.loadURDF('jenga/jenga.urdf', basePosition=[0,-0.05,0+.025*(1)])
		# block_measure = tuple(map(lambda i, j: i - j, pb.getAABB(jengaId)[1], pb.getAABB(jengaId)[0]))
		# print("Block Measure: ", block_measure)

		#block_height = 
		#block_length = 
		self.jengaObject=[]
		fix_flag = False
		for layer in range(self.num_layer): # test:6; total:18
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
		# print("Created the Jenga Tower!")
		# pos, ang = pb.getBasePositionAndOrientation(self.jengaObject[-1], self.physicsClient)
		# print(pos)
		return self.state

if __name__ == "__main__":
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
		action = np.random.choice(env.blocks_buffer)
		print(action)
		state,rw,done,info = env.step(action)
		print(rw)

	# show what happened following
	for i in range(300):
		pb.stepSimulation()
		time.sleep(1./240.)
	# close the pybullet
	pb.disconnect()