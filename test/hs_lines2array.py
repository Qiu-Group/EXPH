import numpy as np

# We want to transfer a set of continuous points to a line:
# e.g.: [[0,1,0],[0,2,0],[0,3,0]...] --> [1,2,3...]

kpt_file = np.loadtxt('kpt.dat')
assert kpt_file.shape[1] == 3
assert kpt_file.shape[0] > 1
print('shape of kpt.dat is legal')