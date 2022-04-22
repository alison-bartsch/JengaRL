import numpy as np


state = np.zeros((54, 54))

for i in np.arange(54)[::3]:
	print(i)
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
	elif i == 51:
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

print(state[4,:])
print(state[:,4])
