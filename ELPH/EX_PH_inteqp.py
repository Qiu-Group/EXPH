import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
# from ELPH.EX_PH_mat import gqQ
from Common.inteqp import interqp_2D
from IO.IO_common import read_kmap, read_lattice
from IO.IO_acv import read_Acv, read_Acv_exciton_energy
from IO.IO_gkk import read_gkk, read_omega
from IO.IO_common import read_bandmap, read_kmap,construct_kmap
from Common.progress import ProgressBar
from Common.common import equivalence_order, equivalence_no_order, move_k_back_to_BZ_1, isDoubleCountK, frac2carte


# These functions offer good linear interpolation for gqQ, omega(q) and OMEGA(Q)

# (2) omega_inteqp_q --> interpolate to a fine q-grid given v (quantum number of phonon)
# (3) OMEGA_inteqp_Q --> interpolate  to a find Q-grid given nS (quantum number of exciton)

# (5) kgrid_inteqp_complete(k_grid_2D):
# (6) dispersion_inteqp_complete(dispersion_2D):
#input

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
        # tododone: use index to find omega instead of directly using it!!!
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

# TODOdone: realize Gamma_itneqp!!




# tododone: add boundary condition !!
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
    # res = omega_inteqp_q(interpo_size=160,new_q_out=True, path='../')
    res = OMEGA_inteqp_Q(interpo_size=4,path='../',new_Q_out=True)


    grid = np.array([res[0].flatten(), res[1].flatten()]).T
    # print('is double count:',isDoubleCountK(grid))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(res[0], res[1], res[2][0], cmap=cm.cool)
    plt.show()

    # res = interpolation_check_for_Gamma_calculation(interpo_size=4,path='../')

    pass