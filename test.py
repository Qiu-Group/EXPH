import numpy as np
import h5py as h5


try:
    kmap = np.loadtxt('kkqQmap.dat')
except:
    raise Exception("failed to open kkqQmap.dat")
res = {}
for i in range(kmap.shape[0]):
    res['  %.6f    %.6f    %.6f'%(kmap[i,0:3][0],kmap[i,0:3][1],kmap[i,0:3][2])] = kmap[i,3:]