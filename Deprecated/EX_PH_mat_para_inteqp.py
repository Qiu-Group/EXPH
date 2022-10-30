import numpy as np
from IO.IO_common import read_bandmap
from IO.IO_acv import read_Acv
from IO.IO_gkk import read_gkk
from IO.IO_common import read_bandmap, read_kmap,construct_kmap
from Common.common import move_k_back_to_BZ_1
from Common.h5_status import check_h5_tree
import h5py as h5
from Common.progress import ProgressBar
from ELPH.EX_PH_inteqp import kgrid_inteqp_complete,dispersion_inteqp_complete
from Common.inteqp import interqp_2D
from ELPH.EX_PH_mat import gqQ


def gqQ_inteqp_get_coarse_grid(n_ex_acv_index=2, m_ex_acv_index=0, v_ph_gkk=3, Q_kmap=15, #interpo_size=12
                               new_q_out=False, acvmat=None, gkkmat=None,kmap=None, kmap_dic=None, bandmap_occ=None,muteProgress=True,
                                 path='./',q_map_start_para='nopara', q_map_end_para='nopara'):
    """
    !!! parallel is over q!!!
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
    :return: |gqQ| interpoated by q: interpo_size * interpo_size
    """
    # Q_kmap_star = 0 # exciton start momentum Q (index)
    # n_ex_acv = 0 # exciton start quantum state S_i (index): we only allow you to set one single state right now
    # m_ex_acv = 1 # exciton final quantum state S_f (index)
    # v_ph_gkk = 3 # phonon mode (index)
    # mute = True
    # interpo_size = 4

    # outfilename = "exciton_phonon_mat_inteqp.dat"
    # interpo_size = interpo_size + 1

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

    # loop over k_kmap

    if q_map_start_para == 'nopara' and q_map_end_para == 'nopara':
        q_map_start_para = 0
        q_map_end_para = kmap.shape[0]
    else:
        if type(q_map_start_para) is int and type(q_map_end_para) is int:
            pass
        else:
            raise Exception("the parallel parameter is not int")


    for q_kmap in range(q_map_start_para,q_map_end_para):
        if not muteProgress:
            progress.current += 1
            progress()
        if new_q_out:
            res[q_kmap,:3] = kmap[q_kmap,:3]
        res[q_kmap,3] = np.abs(gqQ(n_ex_acv_index=n_ex_acv_index,
                                   m_ex_acv_index=m_ex_acv_index,
                                   v_ph_gkk=v_ph_gkk,Q_kmap=Q_kmap,
                                   q_kmap=q_kmap,
                                   acvmat=acvmat,
                                   gkkmat=gkkmat,
                                   kmap=kmap,
                                   kmap_dic=kmap_dic,
                                   bandmap_occ=bandmap_occ,
                                   muteProgress=True
                                   ))
    return res, new_q_out


def gqQ_inteqp_q(res, new_q_out=False, interpo_size=12, kmap=None, path='./'):
    """
    :param res:
        !! res = np.zeros((kmap.shape[0],4)), which is from output of gqQ_inteqp_get_coarse_grid
    :param new_q_out: ..
    :param interpo_size: ..
    :param kmap: ..
    :param path: ..
    :return:
    """

    interpo_size = interpo_size + 1

    if kmap is None:
        kmap = read_kmap(path=path)

    # interpolation for q-grid
    qxx_new = "None"
    qyy_new = "None"
    if new_q_out:
        qxx = res[:,0].reshape((int(np.sqrt(kmap.shape[0])),int(np.sqrt(kmap.shape[0])))) #qx
        qyy = res[:,1].reshape((int(np.sqrt(kmap.shape[0])),int(np.sqrt(kmap.shape[0])))) #qy
        # qzz = res[:,2].reshape((int(np.sqrt(kmap.shape[0])),int(np.sqrt(kmap.shape[0])))) #qz
        #================================================
        #boundary condition
        qxx_temp = kgrid_inteqp_complete(qxx)
        qyy_temp = kgrid_inteqp_complete(qyy)
        # ----------------------------------------------
        qxx_new = interqp_2D(qxx_temp, interpo_size=interpo_size)[:interpo_size-1, :interpo_size-1]
        qyy_new = interqp_2D(qyy_temp, interpo_size=interpo_size)[:interpo_size-1, :interpo_size-1]
        # qzz_new = interqp_2D(qzz, interpo_size=interpo_size)

    # interpolation for result
    resres = res[:,3].reshape((int(np.sqrt(kmap.shape[0])),int(np.sqrt(kmap.shape[0])))) #res
    # ================================================
    # boundary condition
    resres_temp = dispersion_inteqp_complete(resres)
    # ------------------------------------------------
    resres_new = interqp_2D(resres_temp,interpo_size=interpo_size)[:interpo_size-1, :interpo_size-1]

    if new_q_out:
        # if qxx_new == "None" or qyy_new == "None":
        #     raise Exception("qxx_new == None or qyy_new == None")
        return [qxx_new, qyy_new, resres_new]
    else:
        return resres_new # interpolate_size * interpolate_size


    # np.savetxt('qxx_new.dat',qxx_new[:,0])
    # np.savetxt('qyy_new.dat',qyy_new[0,:])

# (2) tododone: inteqp for phonon dispersion omega(q)

if __name__ == "__main__":
    # gqQ(n_ex_acv_index=0, m_ex_acv_index=0, v_ph_gkk=3, Q_kmap=6, q_kmap=12, acvmat=read_Acv(), gkkmat=read_gkk())
    # res = gqQ(n_ex_acv_index=8, m_ex_acv_index=3, v_ph_gkk=2, Q_kmap=3, q_kmap=11,path='../')
    res0, new_q_out = gqQ_inteqp_get_coarse_grid(path='../', new_q_out=False) # res is result from each process!
    # gather all result |
    #                   |
    #                   \/
    res = gqQ_inteqp_q(res=res0, new_q_out=new_q_out, path='../')

