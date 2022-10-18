from IO.IO_common import write_loop
from Common.common import frac2carte
from Common.h5_status import check_h5_tree
import numpy as np
import h5py as h5
from IO.IO_common import read_kmap, read_lattice

# todo: double check!!
S = 0 # index of exciton state
path = '../Acv.h5'
outfilename = 'exciton.dat'

f = h5.File(path,'r')
Seigenval = f['exciton_data/eigenvalues'][()]
Qgrid = f['exciton_header/kpoints/Qpt_coor'][()]
bvec = read_lattice('b','../Acv.h5')

res = np.zeros((Qgrid.shape[0],4))
for iQ in range(Qgrid.shape[0]):
    # progress.current += 1
    # progress()

    # res[iQ, :3] = frac2carte(bvec,Qgrid[iQ])  # give out bohr lattice in reciprocal space
    res[iQ,:3] = Qgrid[iQ]
    res[iQ, 3] = Seigenval[iQ,S]

    write_loop(loop_index=iQ, filename=outfilename, array=res[iQ])

np.savetxt(outfilename, res)
