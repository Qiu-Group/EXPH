import numpy as np
from scipy.io import FortranFile

# This script is used to read epmatq from .epb file and reorder electron-phonon matrix based on Fortran order

prefix = 'bn'

ni = 10 # initial
nj = 10 # final
nk = 36
nq = 36
nmu = 6
n_total = ni*nj*nk*nq*nmu

# TODO: check unit of epmatq, transfer it to [meV]
# TODO: try to read this from different pool and joint them together
f = FortranFile(prefix+'.epb1','r')
ep = f.read_reals(dtype='float')
ep_reshap4complex = ep.reshape((nj*ni*nmu*nk*nq,2))

# These three variables el-ph mat in EPW order
ep_real_epworder = ep_reshap4complex[:,0].reshape((nj,ni,nk,nmu,nq),order='F')
ep_imag_epworder = ep_reshap4complex[:,1].reshape((nj,ni,nk,nmu,nq),order='F')
g2_epworder = ep_imag_epworder**2 + ep_real_epworder**2

# Reshape this to fit order in my EXPH order elph_mat(nq,nk,ni,nj,nmode), this order is actually the same as el-ph matrix on interpolated fine grid
# Well, anyway, here is how I organize the el-ph matrix in a very simple way (this might have some memory issue if el-ph matrix is very large)
# (1) (nj,ni,nk,nmu,nq) --> (ni,nj,nk,nmu,nq)
ep_real = np.swapaxes(ep_real_epworder,0,1)
ep_imag = np.swapaxes(ep_imag_epworder,0,1)
g2 = np.swapaxes(g2_epworder,0,1)
# (2) (ni,nj,nk,nmu,nq) --> (nk,ni,nj,nmu,nq)
ep_real = np.swapaxes(ep_real,1,2)
ep_imag = np.swapaxes(ep_imag,1,2)
g2 = np.swapaxes(g2,1,2)
ep_real = np.swapaxes(ep_real,0,1)
ep_imag = np.swapaxes(ep_imag,0,1)
g2 = np.swapaxes(g2,0,1)
# (2) (nk,ni,nj,nmu,nq) --> (nq,nk,ni,nj,nmu)
ep_real = np.swapaxes(ep_real,3,4)
ep_imag = np.swapaxes(ep_imag,3,4)
g2 = np.swapaxes(g2,3,4)
ep_real = np.swapaxes(ep_real,2,3)
ep_imag = np.swapaxes(ep_imag,2,3)
g2 = np.swapaxes(g2,2,3)
ep_real = np.swapaxes(ep_real,1,2)
ep_imag = np.swapaxes(ep_imag,1,2)
g2 = np.swapaxes(g2,1,2)
ep_real = np.swapaxes(ep_real,0,1)
ep_imag = np.swapaxes(ep_imag,0,1)
g2 = np.swapaxes(g2,0,1)


# save data
# f_phase.shape = (n_total, 2)
f_phase = np.zeros((n_total,2))


f_phase[:,0] = ep_real.reshape(n_total)
f_phase[:,1] = ep_imag.reshape(n_total)
f_nophase = np.sqrt(g2.reshape(n_total))

np.savetxt('elphmat_phase.dat',f_phase)
np.savetxt('elphmat.dat',f_nophase)
