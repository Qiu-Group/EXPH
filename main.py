from IO.IO_gkk import create_gkkh5
from IO.IO_acv import create_acvsh5
from Common.h5_status import check_h5_tree
from Common.kgrid_check import k_grid_summary
from Common.band_check import band_summary
from ELPH.EL_PH_mat import gqQ
from ELPH.EX_Ph_scat import Gamma_scat

if __name__ == "__main__":
    # todo: 0.0 find a way to get all of this
    # setting:
    #==========================================================
    save_path = './save/'
    h5_path = './'
    [nQ, nq, nk, nmode, ni, nj] = [144, 16, 144, 9, 4, 4]
    # ==========================================================

    # main program
    # 1.0 Interface
    # ==========================================================
    create_acvsh5(nQ, save_path + 'acvs.save/')
    create_gkkh5(nq, nk, nmode, ni, nj, save_path + 'gkk.save/')
    # ==========================================================

    # 2.0 K-points/band check
    # kmap create
    # bandmap create
    # ==========================================================
    k_grid_summary()
    band_summary()
    # ==========================================================

    # 3.0 g(Q,q) construction
    gqQ_res = gqQ(n_ex_acv_index=1, m_ex_acv_index=0, v_ph_gkk=4, Q_kmap=6, q_kmap=12)
    gamma_res = Gamma_scat()
