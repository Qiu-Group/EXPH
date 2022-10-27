import numpy as np

from ELPH.EX_PH_mat import gqQ
from Common.inteqp import interqp_2D
from IO.IO_common import read_kmap, read_lattice
from IO.IO_acv import read_Acv
from IO.IO_gkk import read_gkk, read_omega
from IO.IO_common import read_bandmap, read_kmap,construct_kmap
from Common.progress import ProgressBar
from Common.common import frac2carte

#input
def gqQ_inteqp_q(n_ex_acv_index=0, m_ex_acv_index=1, v_ph_gkk=3, Q_kmap=0, interpo_size=4 ,new_q_out=True,
        acvmat=None, gkkmat=None,kmap=None, kmap_dic=None, bandmap_occ=None,muteProgress=False,
        path='./',q_map_start_para='nopara', q_map_end_para='nopara'):
    """
    !!! parallel is not added !!!
    This function construct gnmv(Q,q)
    :param n_ex_acv: index of initial exciton state
    :param m_ex_acv: index of final exciton state
    :param v_ph_gkk: index of phonon mode
    :param Q_kmap: exciton momentum in kmap
    :param q_kmap: phonon momentumB in kmap
    :param acvmat: acv matrix (do not read it every time): False -> no input, read it
    :param gkkmat: gkk matrix (do not read it every time):  False -> no input, read it
    :param kmap: kmap matrix (do not read it every time) -> kmap.shape = (kx,ky,kz,Q, k_acv, q, k_gkk):  False -> no input, read it
    :param kmap_dic: kmap dictionary -> kmap_dic = {'  %.5f    %.5f    %.5f' : [Q, k_acv, q, k_gkk], ...}:  False -> no input, read it
    :param bandmap_occ: [bandmap_matrix, occ]:  False -> no input, read it
    :param path: path of *h5 and *dat
    :param k_map_start_para: the start index of k_map (default: 0)
    :param k_map_end_para= the end index of k_map (default: kmap.shape[0])
    :param muteProgress determine if enable progress report
    :return: |gqQ|. The |gkk| unit is meV, but return here is eV
    """
    # Q_kmap_star = 0 # exciton start momentum Q (index)
    # n_ex_acv = 0 # exciton start quantum state S_i (index): we only allow you to set one single state right now
    # m_ex_acv = 1 # exciton final quantum state S_f (index)
    # v_ph_gkk = 3 # phonon mode (index)
    # mute = True
    # interpo_size = 4

    # outfilename = "exciton_phonon_mat_inteqp.dat"


    if acvmat is None:
        acvmat = read_Acv(path=path)
    if gkkmat is None:
        gkkmat = read_gkk(path=path)
    if kmap is None:
        kmap = read_kmap(path=path)
    if kmap_dic is None:
        kmap_dic = construct_kmap(path=path)
    if bandmap_occ is None:
        bandmap_occ = read_bandmap(path=path)


    progress = ProgressBar(kmap.shape[0], fmt=ProgressBar.FULL)
    res = np.zeros((kmap.shape[0],4))

    for q_kmap in range(kmap.shape[0]):
        if not muteProgress:
            progress.current += 1
            progress()
        if new_q_out:
            res[q_kmap,:3] = kmap[q_kmap,:3]
        res[q_kmap,3] = np.abs(gqQ(n_ex_acv_index=n_ex_acv_index, m_ex_acv_index=m_ex_acv_index,v_ph_gkk=v_ph_gkk,Q_kmap=Q_kmap,q_kmap=q_kmap,
                                                        acvmat=acvmat, gkkmat=gkkmat, kmap=kmap, kmap_dic=kmap_dic, bandmap_occ=bandmap_occ, muteProgress=True))

    # interpolation for q-grid
    if new_q_out:
        qxx = res[:,0].reshape((int(np.sqrt(kmap.shape[0])),int(np.sqrt(kmap.shape[0])))) #qx
        qyy = res[:,1].reshape((int(np.sqrt(kmap.shape[0])),int(np.sqrt(kmap.shape[0])))) #qy
        qzz = res[:,2].reshape((int(np.sqrt(kmap.shape[0])),int(np.sqrt(kmap.shape[0])))) #qz
        qxx_new = interqp_2D(qxx, interpo_size=interpo_size)
        qyy_new = interqp_2D(qyy, interpo_size=interpo_size)
        qzz_new = interqp_2D(qzz, interpo_size=interpo_size)

    # interpolation for result
    resres = res[:,3].reshape((int(np.sqrt(kmap.shape[0])),int(np.sqrt(kmap.shape[0])))) #res
    resres_new = interqp_2D(resres,interpo_size=interpo_size)

    if new_q_out:
        return [qxx_new, qyy_new, qzz_new, resres_new]
    else:
        return resres_new


    # np.savetxt('qxx_new.dat',qxx_new[:,0])
    # np.savetxt('qyy_new.dat',qyy_new[0,:])

# (2) todo: inteqp for phonon dispersion omega(q)
# def omega_inteqp_q
# todo: read omega from gkk (the order of value keeps same as gkk, and unit is meV like this:
#  omega_mat[int(q_gkk_index),int(v_ph_gkk_index_loop)] * 10 ** (-3) # dimension [eV] )


# (3) inteqp for exciton dispersion OMEGA(Q)

if __name__ =="__main__":
    res = gqQ_inteqp_q(path='../',new_q_out=False)