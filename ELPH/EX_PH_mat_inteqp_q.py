import numpy as np

from ELPH.EX_PH_mat import gqQ
from Common.inteqp import interqp_2D
from IO.IO_common import read_kmap, read_lattice
from IO.IO_acv import read_Acv
from IO.IO_gkk import read_gkk
from IO.IO_common import read_bandmap, read_kmap,construct_kmap
from Common.progress import ProgressBar
from Common.common import frac2carte

#input
Q_kmap_star = 0 # exciton start momentum Q (index)
n_ex_acv = 0 # exciton start quantum state S_i (index): we only allow you to set one single state right now
m_ex_acv = 1 # exciton final quantum state S_f (index)
v_ph_gkk = 3 # phonon mode (index)
mute = True
interpo_size = 20

path = '../'
bvec = read_lattice('b', path)
acvmat = read_Acv(path=path)
gkkmat = read_gkk(path=path)
kmap = read_kmap(path=path)
kmap_dic = construct_kmap(path=path)
bandmap_occ = read_bandmap(path=path)
# outfilename = "exciton_phonon_mat_inteqp.dat"

progress = ProgressBar(kmap.shape[0], fmt=ProgressBar.FULL)
res = np.zeros((kmap.shape[0],4))

for q_kmap in range(kmap.shape[0]):
    if not mute:
        progress.current += 1
        progress()
    res[q_kmap,:3] = kmap[q_kmap,:3]
    res[q_kmap,3] = np.abs(gqQ(n_ex_acv_index=n_ex_acv, m_ex_acv_index=m_ex_acv,v_ph_gkk=v_ph_gkk,Q_kmap=Q_kmap_star,q_kmap=q_kmap,
                                                    acvmat=acvmat, gkkmat=gkkmat, kmap=kmap, kmap_dic=kmap_dic, bandmap_occ=bandmap_occ, muteProgress=True))

qxx = res[:,0].reshape((int(np.sqrt(kmap.shape[0])),int(np.sqrt(kmap.shape[0])))) #qx
qyy = res[:,1].reshape((int(np.sqrt(kmap.shape[0])),int(np.sqrt(kmap.shape[0])))) #qy
qzz = res[:,2].reshape((int(np.sqrt(kmap.shape[0])),int(np.sqrt(kmap.shape[0])))) #qz

resres = res[:,3].reshape((int(np.sqrt(kmap.shape[0])),int(np.sqrt(kmap.shape[0])))) #qz


# todo: here is wrong!! check inteqp_2d!!! especially for qxx!!
qxx_new = interqp_2D(qxx,interpo_size=interpo_size)
qyy_new = interqp_2D(qyy,interpo_size=interpo_size)
qzz_new = interqp_2D(qzz,interpo_size=interpo_size)
resres_new = interqp_2D(resres,interpo_size=interpo_size)

np.savetxt('qxx_new.dat',qxx_new[:,0])
np.savetxt('qyy_new.dat',qyy_new[0,:])
