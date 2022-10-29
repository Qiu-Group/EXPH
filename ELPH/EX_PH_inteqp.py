import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from ELPH.EX_PH_mat import gqQ
from Common.inteqp import interqp_2D
from IO.IO_common import read_kmap, read_lattice
from IO.IO_acv import read_Acv, read_Acv_exciton_energy
from IO.IO_gkk import read_gkk, read_omega
from IO.IO_common import read_bandmap, read_kmap,construct_kmap
from Common.progress import ProgressBar
from Common.common import equivalence_order, equivalence_no_order, move_k_back_to_BZ_1, isDoubleCountK, frac2carte


# These functions offer good linear interpolation for gqQ, omega(q) and OMEGA(Q)
# (1) gqQ_inteqp_q  --> interpolate to a fine q-grid given n,m,v,Q
# (2) omega_inteqp_q --> interpolate to a fine q-grid given v (quantum number of phonon)
# (3) OMEGA_inteqp_Q --> interpolate  to a find Q-grid given nS (quantum number of exciton)
# (4) interpolation_check_for_Gamma_calculation(interpo_size, path='./'):
# (5) kgrid_inteqp_complete(k_grid_2D):
# (6) dispersion_inteqp_complete(dispersion_2D):
#input
def gqQ_inteqp_q(n_ex_acv_index=0, m_ex_acv_index=1, v_ph_gkk=3, Q_kmap=0, interpo_size=12 ,new_q_out=False,
        acvmat=None, gkkmat=None,kmap=None, kmap_dic=None, bandmap_occ=None,muteProgress=True,
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
    :return: |gqQ| interpoated by q: interpo_size * interpo_size
    """
    # Q_kmap_star = 0 # exciton start momentum Q (index)
    # n_ex_acv = 0 # exciton start quantum state S_i (index): we only allow you to set one single state right now
    # m_ex_acv = 1 # exciton final quantum state S_f (index)
    # v_ph_gkk = 3 # phonon mode (index)
    # mute = True
    # interpo_size = 4

    # outfilename = "exciton_phonon_mat_inteqp.dat"
    interpo_size = interpo_size + 1

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
    for q_kmap in range(kmap.shape[0]):
        if not muteProgress:
            progress.current += 1
            progress()
        if new_q_out:
            res[q_kmap,:3] = kmap[q_kmap,:3]
        res[q_kmap,3] = np.abs(gqQ(n_ex_acv_index=n_ex_acv_index, m_ex_acv_index=m_ex_acv_index,v_ph_gkk=v_ph_gkk,Q_kmap=Q_kmap,q_kmap=q_kmap,
                                    acvmat=acvmat, gkkmat=gkkmat, kmap=kmap, kmap_dic=kmap_dic, bandmap_occ=bandmap_occ, muteProgress=True
                                   ))

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

# (2) todo: inteqp for phonon dispersion omega(q)
def omega_inteqp_q(interpo_size=12, new_q_out=False,path="./"):
    """
    :param interpo_size: ..
    :param new_q_out: output new q-grid or not?
    :param path: ..
    :return:
    """
    # interpo_size = 36
    # new_q_out = True
    # path = '../'
    interpo_size = interpo_size + 1

    omega_mat = read_omega(path=path) # dimension [meV]
    n_phonon = omega_mat.shape[1]
    kmap = read_kmap(path=path)  # load kmap matrix


    omega_res = np.zeros((n_phonon,interpo_size-1,interpo_size-1))
    for j_phonon in range(n_phonon):
        # print(j_phonon)
        omega_q_index_list = list(map(int,kmap[:,5])) # kmap[:,5] is for q !!!
        # use array as index can have better efficiency!!
        # todo: use index to find omega instead of directly using it!!!
        temp_omega = omega_mat[:,j_phonon][np.array(omega_q_index_list)].reshape((int(np.sqrt(kmap.shape[0])),int(np.sqrt(kmap.shape[0])))) #qx
        # ================================================
        # boundary condition
        omega_res_temp = dispersion_inteqp_complete(temp_omega)
        omega_res[j_phonon] = interqp_2D(omega_res_temp, interpo_size=interpo_size)[:interpo_size-1, :interpo_size-1]

    qxx_new = "None"
    qyy_new = "None"
    if new_q_out:
        qxx = kmap[:, 0].reshape((int(np.sqrt(kmap.shape[0])), int(np.sqrt(kmap.shape[0]))))
        qyy = kmap[:, 1].reshape((int(np.sqrt(kmap.shape[0])), int(np.sqrt(kmap.shape[0]))))
# ================================================
        # boundary condition
        qxx_temp = kgrid_inteqp_complete(qxx)
        qyy_temp = kgrid_inteqp_complete(qyy)
# ----------------------------------------------
        qxx_new = interqp_2D(qxx_temp, interpo_size=interpo_size)[:interpo_size-1, :interpo_size-1]
        qyy_new = interqp_2D(qyy_temp, interpo_size=interpo_size)[:interpo_size-1, :interpo_size-1]

    if new_q_out:
        # if qxx_new == "None" or qyy_new == "None":
        #     raise Exception("qxx_new == None or qyy_new == None")
        return [qxx_new, qyy_new, omega_res]
    else:
        return omega_res
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # # surf = ax.plot_surface(qxx, qyy, omega_mat[:,3].reshape((int(np.sqrt(kmap.shape[0])),int(np.sqrt(kmap.shape[0])))) , cmap=cm.cool)
    # surf = ax.plot_surface(qxx_new, qyy_new, omega_res[3], cmap=cm.cool)
    # plt.show()
# exciton_energy = read_Acv_exciton_energy(path=path)

def OMEGA_inteqp_Q(interpo_size=12, new_Q_out=False, path="./"):
    interpo_size = interpo_size + 1
    # interpo_size = 12
    # new_Q_out = True
    # path = '../'
    exciton_energy = read_Acv_exciton_energy(path=path)
    n_exciton= exciton_energy.shape[1]
    kmap = read_kmap(path=path)  # load kmap matrix

    OMEGA_res = np.zeros((n_exciton, interpo_size-1, interpo_size-1))
    for j_xt in range(n_exciton):
        OMEGA_Q_index_list = list(map(int, kmap[:, 3]))  # kmap[:,5] is for q !!!
        temp_OMEGA = exciton_energy[:, j_xt][np.array(OMEGA_Q_index_list)].reshape((int(np.sqrt(kmap.shape[0])), int(np.sqrt(kmap.shape[0]))))  # qx
        # ================================================
        # boundary condition
        OMEGA_res_temp = dispersion_inteqp_complete(temp_OMEGA)
        OMEGA_res[j_xt] = interqp_2D(OMEGA_res_temp, interpo_size=interpo_size)[:interpo_size-1, :interpo_size-1]


    Qxx_new = "None"
    Qyy_new = "None"
    if new_Q_out:
        Qxx = kmap[:, 0].reshape((int(np.sqrt(kmap.shape[0])), int(np.sqrt(kmap.shape[0]))))
        Qyy = kmap[:, 1].reshape((int(np.sqrt(kmap.shape[0])), int(np.sqrt(kmap.shape[0]))))

        # ================================================
        # boundary condition
        Qxx_temp = kgrid_inteqp_complete(Qxx)
        Qyy_temp = kgrid_inteqp_complete(Qyy)
        # ----------------------------------------------

        Qxx_new = interqp_2D(Qxx_temp, interpo_size=interpo_size)[:interpo_size-1, :interpo_size-1]
        Qyy_new = interqp_2D(Qyy_temp, interpo_size=interpo_size)[:interpo_size-1, :interpo_size-1]

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_surface(qxx, qyy, omega_mat[:,3].reshape((int(np.sqrt(kmap.shape[0])),int(np.sqrt(kmap.shape[0])))) , cmap=cm.cool)
    # surf = ax.plot_surface(Qxx_new, Qyy_new, OMEGA_res[2], cmap=cm.cool)
    # surf = ax.plot_surface(Qxx, Qyy, exciton_energy[:, 2][np.array(OMEGA_Q_index_list)].reshape((int(np.sqrt(kmap.shape[0])), int(np.sqrt(kmap.shape[0])))), cmap=cm.cool)
    # plt.show()

    if new_Q_out:
        return [Qxx_new, Qyy_new, OMEGA_res]
    else:
        return OMEGA_res

# (3) inteqp for exciton dispersion OMEGA(Q)

# TODO: realize Gamma_itneqp!!

def interpolation_check_for_Gamma_calculation(interpo_size, path='./', mute=False):
    """
    WARNING: we only support integer multiple interpolation: k-grid after interpolation could cover k-grid before interpolation
    WARNING: interpolation following such rule (only for 2D):
        coarse gird: n_co * n_co * 1
        fine   grid: n_fi * n_fi * 1 (n_fi = interpo_size)
        n_fi = (n_co - 1) * m + 1, where m is the multiple of coarse grid
    Run this before interpolate any interpolation for Gamma
    :param interpo_size: ..
    :param path: 'kkqQmap.dat', 'Acv.h5', 'gkk.h5'
    :return:
     (0) interpolated q/Q-grid
     (1) Qq_dic: Qq_DIC = {'  %.5f    %.5f    %.5f' : Qq_fine}, where Qq_fine is index of interpolated index in gqQ_interpolated(q), omega(q) and OMEGA(Q)
     (2) interpolated phonon frequency
     (3) interpolated exciton frequency
    """
    kmap = read_kmap(path=path)
    n_co = int(np.sqrt(kmap.shape[0]))
    n_fi = interpo_size
    if (n_fi ) % (n_co ) != 0:
        raise Exception("Only support integer multiple interpolation: k-grid after interpolation should cover k-grid before interpolation (e.g.: (4,4,1) --×10--> (32, 32, 1))")
    else:
        if not mute:
            print("[interpolation size]: check")
    res_gqQ = gqQ_inteqp_q(interpo_size=interpo_size,path=path,new_q_out=True)
    res_omega = omega_inteqp_q(interpo_size=interpo_size, path=path,new_q_out=True)
    res_OMEGA = OMEGA_inteqp_Q(interpo_size=interpo_size,path=path,new_Q_out=True)
    grid_q_gqQ = np.array([res_gqQ[0].flatten(), res_gqQ[1].flatten()]).T
    grid_q_omega = np.array([res_omega[0].flatten(), res_omega[1].flatten()]).T
    grid_q_OMEGA = np.array([res_OMEGA[0].flatten(), res_OMEGA[1].flatten()]).T
    # print("A-E-B?", equivalence_no_order(grid_q_gqQ, grid_q_omega))
    non_equal_count = 0
    # if grid_q_gqQ.shape != res_omega[2].flatten().shape:
    #     non_equal_count += 1
    if not equivalence_order(grid_q_gqQ, grid_q_omega):
        non_equal_count += 1
    if not equivalence_order(grid_q_gqQ, grid_q_OMEGA):
        non_equal_count += 1
    if not equivalence_order(grid_q_omega, grid_q_OMEGA):
        non_equal_count += 1
    if non_equal_count == 0:
        if not mute:
            print("[qQ-grid (interpolated) check]: pass")
            print("interpolated qQ-grid of (%s, %s, 1) are in the same order!"%(interpo_size, interpo_size))
        grid_q_gqQ_res = np.vstack( (grid_q_gqQ.T,np.zeros((grid_q_gqQ.shape[0])).T)).T

        Qq_dic = {}
        for i in range(grid_q_gqQ_res.shape[0]):
            Qq_dic['  %.5f    %.5f    %.5f' % (grid_q_gqQ_res[i, 0:3][0], grid_q_gqQ_res[i, 0:3][1], grid_q_gqQ_res[i, 0:3][2])] = i

        return [grid_q_gqQ_res, Qq_dic, res_omega[2], res_OMEGA[2]]
    else:
        raise Exception("[qQ-grid (interpolated) check]: failed")

#todo: suggestion function for interpolation size
# rewrite Gamma Calculation, write document for kmap, k_dic, Qq_dic

# def Qpoints_2_Qfi_dic_generate(Q_grid, q_grid):
#     pass


# todo: add boundary condition !!
def kgrid_inteqp_complete(k_grid_2D):
    """
    input should be 2D_array k-grid
    :param k_grid_2D: k_grid_2D.shape = (n, n)
    :return: k_grid_2D_boundar.shape = (n+1, n+1)
    """
    size = k_grid_2D.shape[0]
    k_grid_2D_new = np.zeros((size+1,size+1))
    k_grid_2D_new[:size,:size] = k_grid_2D
    if k_grid_2D[0,0] != 0:
        raise Exception("k_grid[0,0] should start with 0!, check k_grid before interpolation!!")
    if k_grid_2D[0,-1] == 0:
        delta = k_grid_2D[1,0] - k_grid_2D[0,0]
        # print(k_grid_2D_new.shape, k_grid_2D)
        k_grid_2D_new[size,:size] = np.ones((size)) * (delta + k_grid_2D_new[size-1,0])
        k_grid_2D_new[:,size] = k_grid_2D_new[:,0]
    else:
        delta = k_grid_2D[0,1] - k_grid_2D[0,0]
        # print(k_grid_2D_new)
        # print(k_grid_2D)
        k_grid_2D_new[:size,size] = np.ones((size)) * (delta + k_grid_2D_new[0,size-1])
        k_grid_2D_new[size,:] = k_grid_2D_new[0,:]

    return k_grid_2D_new

def dispersion_inteqp_complete(dispersion_2D):
    """
    input should be 2D_array dispersion
    :param k_grid_2D: k_grid_2D.shape = (n, n)
    :return: k_grid_2D_boundar.shape = (n+1, n+1)
    """
    size = dispersion_2D.shape[0]
    dispersion_2D_new = np.zeros((size+1,size+1))
    dispersion_2D_new[:size,:size] = dispersion_2D

    dispersion_2D_new[:size,size] = dispersion_2D_new[:size,0]
    dispersion_2D_new[size,:] = dispersion_2D_new[0,:]

    return dispersion_2D_new

if __name__ =="__main__":
    # res = gqQ_inteqp_q(path='../',new_q_out=True,interpo_size=24)
    # res = omega_inteqp_q(interpo_size=4,new_q_out=True, path='../')
    res = OMEGA_inteqp_Q(interpo_size=160,path='../',new_Q_out=True)


    grid = np.array([res[0].flatten(), res[1].flatten()]).T
    # print('is double count:',isDoubleCountK(grid))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(res[0], res[1], res[2][2], cmap=cm.cool)
    plt.show()

    # res = interpolation_check_for_Gamma_calculation(interpo_size=4,path='../')

    pass