import numpy as np
from Common.distribution import BE, FD, Dirac_1, Dirac_2
from IO.IO_gkk import read_omega, read_gkk
from IO.IO_acv import read_Acv, read_Acv_exciton_energy
from IO.IO_common import read_bandmap, read_kmap, construct_kmap
from ELPH.EX_PH_mat import gqQ
from Common.progress import ProgressBar
from Common.common import move_k_back_to_BZ_1
import time
# calculate the scattering rate
# omega.shape  = (nq, nmode)
# exciton_energy.shape = (nQ, nS)
# acvmat.shape = (nQ,nS,nk,nc,nv,2)
# gkkmat.shape = (nq,nk,ni,nj,nmode)
# gqQ(n_ex_acv_index, m_ex_acv_index, v_ph_gkk, Q_kmap, q_kmap,
#                  acvmat, gkkmat, kmap, kmap_dic, bandmap_occ,muteProgress)


#==============================================================================================================>>>>>>>
def Gamma_scat(Q_kmap=6, n_ext_acv_index=0,T=100, degaussian=0.001,
               muteProgress=False, path='./',q_map_start_para='nopara', q_map_end_para='nopara'):
    """
    !!! parallel over q_kmap !!!
    !!! PARALLEL OVER q_KMAP !!!
    :param Q_kmap: exciton momentum in kmap
    :param n_ext_acv_index: index of initial exciton state
    :param T: temperature
    :param degaussian: sigma in Gaussian distribution
    :return: equation (6) in Bernardi's paper
    """

    # (1) input
    # Q_kmap = 6 # this is index of kmap
    # n_ext_acv_index = 0 # this is index of exciton state in Acv
    # T = 100 # randomly choosing
    # degaussian = 0.001

    # (2) load and construct map:
    acvmat = read_Acv(path=path) # load acv matrix
    gkkmat =read_gkk(path=path) # load gkk matrix
    kmap = read_kmap(path=path)  # load kmap matrix
    [bandmap, occ] = read_bandmap(path=path)  # load band map and number of occupation
    kmap_dic = construct_kmap(path=path)  # construct kmap dictionary {'k1 k2 k3':[0 0 0 0]}: this is used for mapping final state of scattering
    omega_mat = read_omega(path=path) # dimension [meV]
    exciton_energy = read_Acv_exciton_energy(path=path)
    h_bar = 6.582119569E-16     # dimension = [eV.s]
    # number of point, band and phonon mode
    nc = bandmap[:, 0][-1] - occ  # load number of conduction band
    nv = occ + 1 - bandmap[:, 0][0]  # load number of valence band
    n_phonon = omega_mat.shape[1]
    Nqpt = kmap.shape[0] # number of q-points


    # (3) calculate scattering rate
    # note: kmap.shape(nk, information=(kx,ky,kz,Q, k_acv, q, k_gkk))
    if not muteProgress:
        print('\n[Exciton Scattering]: n=', n_ext_acv_index, ' Q=', Q_kmap, 'T=',T)
        progress = ProgressBar(kmap.shape[0], fmt=ProgressBar.FULL) # progress


    # loop start with q
    # initialize for loop
    collect = []
    # it seems that we can directly add first and second together, so we don't need Gamma_res
    Gamma_res = 0
    # Since Gamma_first and Gamma_second don't share same renormalization factor, so we need to split them

    Gamma_first_res = 0
    Gamma_second_res = 0
    factor = 2*np.pi/(h_bar*Nqpt) # dimension = [eV-1.s-1]
    # Since Gamma_first and Gamma_second don't share same renormalization factor, so we need to split them
    dirac_normalize_factor_first = 0.0
    dirac_normalize_factor_second = 0.0

    #=============================
    # todo: double check if this is right (parallel unit):
    if q_map_start_para == 'nopara' and q_map_end_para == 'nopara':
        status_for_para = 'nopara'
        q_map_start_para = 0
        q_map_end_para = kmap.shape[0]
    else:
        status_for_para = 'para'
        if type(q_map_start_para) is int and type(q_map_end_para) is int:
            pass
        else:
            raise Exception("the parallel parameter is not int")

    for q_kmap in range(q_map_start_para, q_map_end_para):  # q_kmap is the index of kmap from point 0-15 in kmap.dat (e.g)
        if not muteProgress:
            progress.current += 1
            progress()
        for m_ext_acv_index_loop in range(exciton_energy.shape[1]): # loop over initial exciton state m
            for v_ph_gkk_index_loop in range(n_phonon): # loop over phonon mode v

                # (1) ex-ph matrix
                # gqQ(n_ex_acv_index, m_ex_acv_index, v_ph_gkk, Q_kmap, q_kmap,
                #                  acvmat, gkkmat, kmap, kmap_dic, bandmap_occ,muteProgress)
                #===============================================
                # t_s = time.time()

                gqQ_sq_temp = np.abs(gqQ(n_ex_acv_index=n_ext_acv_index,
                           m_ex_acv_index=m_ext_acv_index_loop,
                           v_ph_gkk= v_ph_gkk_index_loop,
                           Q_kmap=Q_kmap,
                           q_kmap=q_kmap,
                           acvmat=acvmat,
                           gkkmat=gkkmat,
                           kmap=kmap,
                           kmap_dic=kmap_dic,
                           bandmap_occ= [bandmap,occ],
                           muteProgress=True
                           ))**2 # dimension [eV^2]
                # ===============================================
                # only for test
                # gqQ_sq_temp = 1.564949**2
                # t_e2 = time.time()

                # print('te1 - ts:', t_e1 - t_s)
                # print("te2 - ts:", t_e2 - t_s)
                # q_gkk and Q+q_acv index
                # kmapout[x] = [Q, k_acv, q, k_gkk]


                # print('gqQ_sq_temp is', gqQ_sq_temp)
                # print("gqQ_sq_temp == 0",gqQ_sq_temp == 0)

                # Skip if q = 0 and nmode = [0,1,2] <- longwave limit
                if gqQ_sq_temp == 0:
                    continue


                [Q_acv_index, q_gkk_index] = [kmap[Q_kmap, 3], kmap[q_kmap, 5]]
                Q_plus_q_point = move_k_back_to_BZ_1(kmap[Q_kmap, 0:3] + kmap[q_kmap, 0:3])
                key_temp = '  %.5f    %.5f    %.5f' % (Q_plus_q_point[0], Q_plus_q_point[1], Q_plus_q_point[2])
                Q_plus_q_kmapout = kmap_dic[key_temp.replace('_','')]
                Qpr_as_Q_plus_q_acv_index = Q_plus_q_kmapout[0]

                # find energy for exciton and phonon (you should use index_acv and index_gkk)
                # omega.shape  = (nq, nmode)
                # exciton_energy.shape = (nQ, nS)
                # OMEGA_xx has included h_bar

                # todo Check this !!!
                omega_v_q_temp     = omega_mat[int(q_gkk_index),int(v_ph_gkk_index_loop)] * 10 ** (-3) # dimension [eV]
                OMEGA_m_Q_plus_q_temp = exciton_energy[int(Qpr_as_Q_plus_q_acv_index), int(m_ext_acv_index_loop)] # dimension [eV]
                OMEGA_n_Q_temp        = exciton_energy[int(Q_acv_index),               int(m_ext_acv_index_loop)] # dimension [eV]

                # (2) left part
                # print(OMEGA_m_Q_plus_q_temp)
                # tododone: check warning: RuntimeWarning: divide by zero encountered!!!
                # print(BE(omega=OMEGA_m_Q_plus_q_temp, T=T))
                # print(OMEGA_m_Q_plus_q_temp)
                # tododone: Dirac normalization!!!!!
                # Here, Dirac 1 is Gaussian; Dirac 2 is Square wave
                # both of Dirac 1 and Dirac 2 should be normalized

                # here is the normalize factor:
                # dirac_normalize_factor_first = dirac_normalize_factor_first + Dirac_1(OMEGA_n_Q_temp - OMEGA_m_Q_plus_q_temp - omega_v_q_temp, sigma=degaussian)
                # dirac_normalize_factor_second = dirac_normalize_factor_second + Dirac_1(OMEGA_n_Q_temp - OMEGA_m_Q_plus_q_temp + omega_v_q_temp, sigma=degaussian)



                distribution_first_temp = (
                                    (BE(omega=omega_v_q_temp, T=T) + 1 + BE(omega=OMEGA_m_Q_plus_q_temp, T=T))
                                     * Dirac_1(OMEGA_n_Q_temp - OMEGA_m_Q_plus_q_temp - omega_v_q_temp, sigma=degaussian))


                distribution_second_temp = (
                                    (BE(omega=omega_v_q_temp, T=T) - BE(omega=OMEGA_m_Q_plus_q_temp, T=T))
                                    * Dirac_1(OMEGA_n_Q_temp - OMEGA_m_Q_plus_q_temp + omega_v_q_temp, sigma=degaussian)

                                     )
                # Since Gamma_first and Gamma_second don't share same renormalization factor, so we need to split them
                Gamma_first_res = Gamma_first_res + factor * gqQ_sq_temp * distribution_first_temp
                Gamma_second_res = Gamma_second_res + factor * gqQ_sq_temp * distribution_second_temp

    # print('1st factor:',dirac_normalize_factor_first)
    # print('2nd factor:',dirac_normalize_factor_second)
    # return Gamma_res
    # IF this is a series test:
    # print(status_for_para)

    # TODO: Discuss with Diana!!!

    if status_for_para == 'nopara':
        # return Gamma_first_res / dirac_normalize_factor_first + Gamma_second_res / dirac_normalize_factor_second
        # Warning: gamma should not be normalized
        return Gamma_first_res + Gamma_second_res
    # (a) w/ normalization
    # return Gamma_first_res/dirac_normalize_factor_first + Gamma_second_res/dirac_normalize_factor_second
    # Warning: normalization factor should be added at the last step
    else:
        # return  [Gamma_first_res, Gamma_second_res,dirac_normalize_factor_first,dirac_normalize_factor_second]
        # Warning: gamma should not be normalized
        # return [Gamma_first_res, Gamma_second_res,1,1]
        return Gamma_first_res + Gamma_second_res

    # (b) w/o normalization deprecated debug:
    # TODOdone: TODOdone: TODOdone: do not use this!!
    # return Gamma_first_res + Gamma_second_res
#==============================================================================================================>>>>>>>








if __name__ == "__main__":
    res = Gamma_scat(Q_kmap=15, n_ext_acv_index=2,T=100, degaussian=0.001,path='../')





