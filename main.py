from IO.IO_gkk import create_gkkh5
from IO.IO_acv import create_acvsh5
from Common.h5_status import check_h5_tree
from Common.kgrid_check import k_grid_summary
from Common.band_check import band_summary
from ELPH.EX_PH_mat import gqQ
from ELPH.EX_PH_scat import Gamma_scat_test_nointeqp
from ELPH.EX_PH_lifetime_all_Q import Exciton_Life
from Common.common import frac2carte
import numpy as np
import h5py as h5

if __name__ == "__main__":
    # todo: 0.0 find a way to get all of this
    # 0.0 setting:
    #==========================================================
    save_path = './save_441/'
    h5_path = './'
    [nQ, nq, nk, nmode, ni, nj] = [144, 16, 144, 9, 4, 4]
    # ==========================================================


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
    # gqQ_res = gqQ(n_ex_acv_index=8, m_ex_acv_index=3, v_ph_gkk=2, Q_kmap=3, q_kmap=11)
    # gamma_res = Gamma_scat(Q_kmap=15, n_ext_acv_index=2,T=100, degaussian=0.001,path='./')
    # Exciton_Life()

