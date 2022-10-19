from IO.IO_common import write_loop
from Common.common import frac2carte
from Common.h5_status import check_h5_tree
from ELPH.EL_PH_mat import gqQ
from IO.IO_common import read_kmap, read_lattice
from IO.IO_acv import read_Acv
from IO.IO_gkk import read_gkk
from IO.IO_common import read_bandmap, read_kmap,construct_kmap
from Common.progress import ProgressBar


import numpy as np
import h5py as h5

# def gqQ(n_ex_acv_index=0, m_ex_acv_index=0, v_ph_gkk=3, Q_kmap=6, q_kmap=12,
#          acvmat=None, gkkmat=None, kmap=None, kmap_dic=None, bandmap_occ=None, muteProgress=False):
#     """
#     This function construct gnmv(Q,q)
#     :param n_ex_acv: index of initial exciton state
#     :param m_ex_acv: index of final exciton state
#     :param v_ph_gkk: index of phonon mode
#     :param Q_kmap: exciton momentum in kmap
#     :param q_kmap: phonon momentumB in kmap
#     :param acvmat: acv matrix (do not read it every time): False -> no input, read it
#     :param gkkmat: gkk matrix (do not read it every time):  False -> no input, read it
#     :param kmap: kmap matrix (do not read it every time) -> kmap.shape = (kx,ky,kz,Q, k_acv, q, k_gkk):  False -> no input, read it
#     :param kmap_dic: kmap dictionary -> kmap_dic = {'  %.5f    %.5f    %.5f' : [Q, k_acv, q, k_gkk], ...}:  False -> no input, read it
#     :param bandmap_occ: [bandmap_matrix, occ]:  False -> no input, read it
#     :return: the gkk unit is meV, but return here is eV
#     """

# :param
Q_kmap_star = 0 # exciton start momentum Q (index)
n_ex_acv = 0 # exciton start quantum state S_i (index): we only allow you to set one single state right now
m_ex_acv = [0] # exciton final quantum state S_f (index)
v_ph_gkk = [0,1,2,3,4,5,6,7,8] # phonon mode (index)

path = '../'

bvec = read_lattice('v',path+'Acv.h5')


acvmat = read_Acv(path=path+'Acv.h5')
gkkmat = read_gkk(path=path+'gkk.h5')
kmap = read_kmap(path=path+'kkqQmap.dat')
kmap_dic = construct_kmap(path=path+'kkqQmap.dat')
bandmap_occ = read_bandmap(path=path+'bandmap.dat')
outfilename = "exciton_phonon_mat.dat"

# mapping
# Q_acv_index = kmap[Q_kmap_star, 3]

progress = ProgressBar(kmap.shape[0], fmt=ProgressBar.FULL)

res = np.zeros((kmap.shape[0],4))
for q_kmap in range(kmap.shape[0]):
    progress.current += 1
    progress()
    # q_gkk_index = kmap[q_kmap, 5]
    # print(Q_acv_index)
    # print(q_gkk_index)
    res[q_kmap, :3] = frac2carte(bvec, kmap[q_kmap,:3])
    # res[q_kmap,:3] = kmap[q_kmap,:3]
    for j_final_S in m_ex_acv:
        for j_phonon in v_ph_gkk:
            res[q_kmap,3] = res[q_kmap,3] + np.abs(gqQ(n_ex_acv_index=n_ex_acv, m_ex_acv_index=j_final_S,v_ph_gkk=j_phonon,Q_kmap=Q_kmap_star,q_kmap=q_kmap,
                                                acvmat=acvmat, gkkmat=gkkmat, kmap=kmap, kmap_dic=kmap_dic, bandmap_occ=bandmap_occ, muteProgress=True))


np.savetxt(outfilename, res)

