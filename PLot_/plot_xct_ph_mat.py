from IO.IO_common import write_loop
from Common.common import frac2carte
from Common.h5_status import check_h5_tree
from ELPH.EX_PH_mat import gqQ, gqQ_inteqp_q_nopara
from IO.IO_common import read_kmap, read_lattice
from IO.IO_acv import read_Acv
from IO.IO_gkk import read_gkk
from IO.IO_common import read_bandmap, read_kmap,construct_kmap
from Common.progress import ProgressBar
from IO.IO_common import read_kmap, read_lattice
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import sys

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

def plot_ex_ph_mat_inteqp(Q_kmap_star=0, n_ex_acv=0, m_ex_acv=[0,1,2,3],v_ph_gkk=[0,1,2,3,4,5,6,7,8], interposize = 4,path='./',mute=True,outfilename = "exciton_phonon_mat.dat"):
    bvec = read_lattice('b',path)
    acvmat = read_Acv(path=path)
    gkkmat = read_gkk(path=path)
    kmap = read_kmap(path=path)
    kmap_dic = construct_kmap(path=path)
    bandmap_occ = read_bandmap(path=path)
    size = int(interposize ** 2)

    count = 0
    res = np.zeros((size,4))
    progress = ProgressBar(len(v_ph_gkk), fmt=ProgressBar.FULL)

    for j_phonon in v_ph_gkk:
        if not mute:
            progress.current += 1
            progress()
            sys.stdout.flush()
        for j_final_S in m_ex_acv:

            if count == 0:
                [qxx_new, qyy_new, resres_new] = gqQ_inteqp_q_nopara(n_ex_acv_index=n_ex_acv,
                                                                     m_ex_acv_index=j_final_S,
                                                                     v_ph_gkk=j_phonon,
                                                                     Q_kmap=Q_kmap_star,
                                                                     interpo_size=interposize,
                                                                     new_q_out=True,
                                                                     acvmat=acvmat,
                                                                     gkkmat=gkkmat,
                                                                     kmap=kmap,
                                                                     kmap_dic=kmap_dic,
                                                                     bandmap_occ=bandmap_occ,
                                                                     muteProgress=True,
                                                                     path=path) # |gqQ|
                res = np.zeros((size, 4))
                # print(qxx_new)
                res[:, 0] = qxx_new.flatten()
                res[:, 1] = qyy_new.flatten()
                res[:, 3] = resres_new.flatten()
                res[:, :3] = frac2carte(bvec, res[:, :3])
                count += 1
            else:
                temp_res = gqQ_inteqp_q_nopara(n_ex_acv_index=n_ex_acv,
                                                                     m_ex_acv_index=j_final_S,
                                                                     v_ph_gkk=j_phonon,
                                                                     Q_kmap=Q_kmap_star,
                                                                     interpo_size=interposize,
                                                                     new_q_out=False,
                                                                     acvmat=acvmat,
                                                                     gkkmat=gkkmat,
                                                                     kmap=kmap,
                                                                     kmap_dic=kmap_dic,
                                                                     bandmap_occ=bandmap_occ,
                                                                     muteProgress=True,
                                                                     path=path) # |gqQ|
                res[:, 3] = res[:, 3] + temp_res.flatten()

    pass
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    np.savetxt(outfilename, res)
    # surf = ax.plot_surface(res[:,0].reshape((interposize,interposize)), res[:,1].reshape((interposize,interposize)), res[:,3].reshape((interposize,interposize)), cmap=cm.cool)
    plt.contourf(res[:,0].reshape((interposize,interposize)), res[:,1].reshape((interposize,interposize)), res[:,3].reshape((interposize,interposize)))
    cbar = plt.colorbar()
    cbar.solids.set_edgecolor("face")
    plt.show()
    np.savetxt(outfilename, res)



# warning: it seems path is not included in gqQ
def plot_ex_ph_mat_nointeqp(Q_kmap_star=0, n_ex_acv=0, m_ex_acv=[0,1,2,3],v_ph_gkk=[0,1,2,3,4,5,6,7,8],path='./',mute=True):
    # Q_kmap_star = 0 # exciton start momentum Q (index)
    # n_ex_acv = 0 # exciton start quantum state S_i (index): we only allow you to set one single state right now
    # m_ex_acv = [0,1,2,3] # exciton final quantum state S_f (index)
    # v_ph_gkk = [0,1,2,3,4,5,6,7,8] # phonon mode (index)

    # path = '../'

    bvec = read_lattice('b',path)


    acvmat = read_Acv(path=path)
    gkkmat = read_gkk(path=path)
    kmap = read_kmap(path=path)
    kmap_dic = construct_kmap(path=path)
    bandmap_occ = read_bandmap(path=path)
    outfilename = "exciton_phonon_mat.dat"

    # mapping
    # Q_acv_index = kmap[Q_kmap_star, 3]

    progress = ProgressBar(kmap.shape[0], fmt=ProgressBar.FULL)

    res = np.zeros((kmap.shape[0],4))
    for q_kmap in range(kmap.shape[0]):
        if not mute:
            progress.current += 1
            progress()
        # q_gkk_index = kmap[q_kmap, 5]
        # print(Q_acv_index)
        # print(q_gkk_index)
        res[q_kmap, :3] = frac2carte(bvec, kmap[q_kmap,:3])
        # res[q_kmap,:3] = kmap[q_kmap,:3]
        for j_final_S in m_ex_acv:
            for j_phonon in v_ph_gkk:
                # print("q_kmap:",q_kmap,' j_final_S:',j_final_S,' j_phonon:',j_phonon)
                res[q_kmap,3] = res[q_kmap,3] + np.abs(gqQ(n_ex_acv_index=n_ex_acv, m_ex_acv_index=j_final_S,v_ph_gkk=j_phonon,Q_kmap=Q_kmap_star,q_kmap=q_kmap,
                                                    acvmat=acvmat, gkkmat=gkkmat, kmap=kmap, kmap_dic=kmap_dic, bandmap_occ=bandmap_occ, muteProgress=True))


    np.savetxt(outfilename, res)
    print('done')

if __name__=="__main__":
    # plot_ex_ph_mat(mute=False, path='../')
    # plot_ex_ph_mat_nointeqp(Q_kmap_star=0, n_ex_acv=0, m_ex_acv=[1],v_ph_gkk=[3],mute=False, path='../')
    plot_ex_ph_mat_inteqp(Q_kmap_star=0, n_ex_acv=2, m_ex_acv=[0,1,2],v_ph_gkk=[3],mute=False, path='../', interposize=4)