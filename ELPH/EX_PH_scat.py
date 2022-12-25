import numpy as np
from Common.distribution import BE, FD, Dirac_1, Dirac_2
from IO.IO_gkk import read_omega, read_gkk
from IO.IO_acv import read_Acv, read_Acv_exciton_energy
from IO.IO_common import read_bandmap, read_kmap, construct_kmap
from ELPH.EX_PH_mat import gqQ, gqQ_inteqp_q_series, gqQ_inteqp_get_coarse_grid, gqQ_inteqp_q_nopara
from Common.progress import ProgressBar
from Common.common import move_k_back_to_BZ_1, equivalence_order
from ELPH.EX_PH_inteqp import omega_inteqp_q,OMEGA_inteqp_Q


import time
# calculate the scattering rate
# omega.shape  = (nq, nmode)
# exciton_energy.shape = (nQ, nS)
# acvmat.shape = (nQ,nS,nk,nc,nv,2)
# gkkmat.shape = (nq,nk,ni,nj,nmode)
# gqQ(n_ex_acv_index, m_ex_acv_index, v_ph_gkk, Q_kmap, q_kmap,
#                  acvmat, gkkmat, kmap, kmap_dic, bandmap_occ,muteProgress)

# (1) Gamma_scat_test_nointeqp --> no interpoaltion
# --> (2) Gamma_scat_low_efficiency --> w/ interpolation and it is a parallel job function (do not use it!), but this is good for series. For para_Gamma_inteqp: see Para_EX_PH_scat
# (3) interpolation_check_for_Gamma_calculation(interpo_size, path='./') --> run it before scattering calculation

#==============================================================================================================>>>>>>>
def Gamma_scat_test_nointeqp(Q_kmap=6, n_ext_acv_index=0,T=100, degaussian=0.001,
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
    # print('exciton_energy.shape[1]',exciton_energy.shape[1])
    #=============================
    # tododone: double check if this is right (parallel unit):
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

    skip_count = 0 # for debug
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
                # print('gqQ_sq_temp:',gqQ_sq_temp)
                # print('gqQ_sq_temp is 0?',gqQ_sq_temp is 0)
                # print('type(gqQ_sq_temp is 0?)',type(gqQ_sq_temp))
                if gqQ_sq_temp == 0:
                    skip_count += 1
                    # print(q_kmap,v_ph_gkk_index_loop)
                    # This is for efficiency and convergence
                    continue


                [Q_acv_index, q_gkk_index] = [kmap[Q_kmap, 3], kmap[q_kmap, 5]]
                Q_plus_q_point = move_k_back_to_BZ_1(kmap[Q_kmap, 0:3] + kmap[q_kmap, 0:3])
                key_temp = '  %.5f    %.5f    %.5f' % (Q_plus_q_point[0], Q_plus_q_point[1], Q_plus_q_point[2])
                # print(key_temp)
                Q_plus_q_kmapout = kmap_dic[key_temp.replace('_','')]
                Qpr_as_Q_plus_q_acv_index = Q_plus_q_kmapout[0]

                # find energy for exciton and phonon (you should use index_acv and index_gkk)
                # omega.shape  = (nq, nmode)
                # exciton_energy.shape = (nQ, nS)
                # OMEGA_xx has included h_bar

                # tododone Check this !!!
                omega_v_q_temp     = omega_mat[int(q_gkk_index),int(v_ph_gkk_index_loop)] * 10 ** (-3) # dimension [eV]
                OMEGA_m_Q_plus_q_temp = exciton_energy[int(Qpr_as_Q_plus_q_acv_index), int(m_ext_acv_index_loop)] # dimension [eV]
                OMEGA_n_Q_temp        = exciton_energy[int(Q_acv_index),               int(n_ext_acv_index)] # dimension [eV]

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

                # print("distribution_first_temp", distribution_first_temp)
                # print("BE part:", (BE(omega=omega_v_q_temp, T=T) + 1 + BE(omega=OMEGA_m_Q_plus_q_temp, T=T)))
                # print("DIrac:", Dirac_1(OMEGA_n_Q_temp - OMEGA_m_Q_plus_q_temp - omega_v_q_temp, sigma=degaussian))
                # print("dirac energy",OMEGA_n_Q_temp - OMEGA_m_Q_plus_q_temp - omega_v_q_temp)
                # print("\n")


    # print('1st factor:',dirac_normalize_factor_first)
    # print('2nd factor:',dirac_normalize_factor_second)
    # return Gamma_res
    # IF this is a series test:
    # print(status_for_para)

    # TODOdone: Discuss with Diana!!!
    progress.done()
    # print('skip %s times out of %s * %s * %s'%(skip_count,q_map_end_para-q_map_start_para,exciton_energy.shape[1], n_phonon))

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



#==============================================================================================================>>>>>>>
def Gamma_scat_low_efficiency_inteqp(Q_kmap=6, n_ext_acv_index=0,T=100, degaussian=0.001, interposize=4, interpolation_check_res = None,
               muteProgress=False, path='./',q_map_start_para='nopara', q_map_end_para='nopara'):
    """
    !!! this is not a job parallel function in reality, but this is good enough for test!!!
    !!! this is a low-effienciency job parallel function!!!
    :param interposize: interpo_size
    :param int_che_res: if None: grid_q_gqQ_res, Qq_dic, res_omega, res_OMEGA will be read by this function
                        else: these parameters wiil be pass by outer loop! (lifetime)
    :param Q_kmap: exciton momentum in kmap
    :param n_ext_acv_index: index of initial exciton state
    :param T: temperature
    :param degaussian: sigma in Gaussian distribution
    :return: equation (6) in Bernardi's paper
    """

    if not interpolation_check_res:
        # this will take longer life, so when calculating lifetime, please pass int_check_result to this function out of Q loop
        # (parameters below is noe depending on specific Q point)
        # interposize is from reading parameter
        # print(interposize)
        [grid_q_gqQ_res, Qq_dic, res_omega, res_OMEGA] = interpolation_check_for_Gamma_calculation(interpo_size=interposize,path=path,mute=muteProgress)
    else:
        # this function get parameter outside
        # interposize is from interpolation_check_res
        [grid_q_gqQ_res, Qq_dic, res_omega, res_OMEGA] = interpolation_check_res
        interposize = int(np.sqrt(interpolation_check_res[0].shape[0]))
        # print('interposize:',interposize)
        # print(grid_q_gqQ_res)
        # print(Qq_dic)
        # print(res_omega)
        # print(res_OMEGA)
            # raise Exception('interposize**2 != int_che_res[0].shape[0]')


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
    # Nqpt = kmap.shape[0] # number of q-points
    Nqpt = interposize**2

    if interposize**2 != kmap.shape[0]:
        if not muteProgress:
            print("[interpolation] yes")
            print(" (%s, %s, 1) -> (%s, %s, 1)"%(int(np.sqrt(kmap.shape[0])),int(np.sqrt(kmap.shape[0])),interposize,interposize))
    else:
        if not muteProgress:
            print("[interpolation] no")

    # (3) calculate scattering rate
    # note: kmap.shape(nk, information=(kx,ky,kz,Q, k_acv, q, k_gkk))
    if not muteProgress:
        print('\n[Exciton Scattering]: n=', n_ext_acv_index, ' Q=', Q_kmap, 'T=',T)
        progress = ProgressBar(exciton_energy.shape[1], fmt=ProgressBar.FULL) # progress

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
    # tododone: double check if this is right (parallel unit):
    if q_map_start_para == 'nopara' and q_map_end_para == 'nopara':
        status_for_para = 'nopara'
        q_map_start_para = 0
        q_map_end_para = grid_q_gqQ_res.shape[0] # loop over interpolated q point
    else:
        status_for_para = 'para'
        if type(q_map_start_para) is int and type(q_map_end_para) is int:
            pass
        else:
            raise Exception("the parallel parameter is not int")

    for m_ext_acv_index_loop in range(exciton_energy.shape[1]):  # loop over initial exciton state m
        if not muteProgress:
            progress.current += 1
            progress()

        for v_ph_gkk_index_loop in range(n_phonon): # loop over phonon mode v

            # get gqQ_inteqp before loop of q
            # Therefore, no q_kmap needed in this equation!!!
            # TODO: parallel over this!!!! write a function: para_gqQ_inteqp_()!!
            # TODO: this step takes the longest time!!!
            #  replace this with new gqQ_inteqp_q, which can realize parallel, it seems Gamma_Scat can only be a series function since it can not achieve
            #  for this part: see EX_PH_mat_para_inteqp.py 150-154 lines
            # we need to define another Gamma_scat_inteqp_q in parallel
            gqQ_sq_inteqp_temp = np.abs(gqQ_inteqp_q_nopara(n_ex_acv_index=n_ext_acv_index,
                                              m_ex_acv_index=m_ext_acv_index_loop,
                                              v_ph_gkk=v_ph_gkk_index_loop,
                                              Q_kmap=Q_kmap, #!!! this Q_kmap from function parameter
                                              interpo_size=interposize,
                                              new_q_out=False,
                                              acvmat=acvmat,
                                              gkkmat=gkkmat,
                                              kmap=kmap,
                                              kmap_dic=kmap_dic,
                                              bandmap_occ= [bandmap,occ],
                                              muteProgress=True,
                                              ))**2 # unit [eV^2]

            # gqQ_sq_inteqp.shape = (interpolate_size, interpolate_size)
            # gqQ_sq_inteqp.flatten.shape = (interpolate_size**2, 1)
            # same order as grid_q_gqQ_res
            gqQ_sq_inteqp_temp = gqQ_sq_inteqp_temp.flatten()


            for q_qQmap in range(q_map_start_para, q_map_end_para):  # q_kmap is the index of kmap from point 0-15 in kmap.dat (e.g)


                # (1) ex-ph matrix
                # gqQ(n_ex_acv_index, m_ex_acv_index, v_ph_gkk, Q_kmap, q_kmap,
                #                  acvmat, gkkmat, kmap, kmap_dic, bandmap_occ,muteProgress)
                #===============================================
                # t_s = time.time()

                # gqQ_sq_temp = np.abs(gqQ(n_ex_acv_index=n_ext_acv_index,
                #            m_ex_acv_index=m_ext_acv_index_loop,
                #            v_ph_gkk= v_ph_gkk_index_loop,
                #            Q_kmap=Q_kmap,
                #            q_kmap=q_kmap,
                #            acvmat=acvmat,
                #            gkkmat=gkkmat,
                #            kmap=kmap,
                #            kmap_dic=kmap_dic,
                #            bandmap_occ= [bandmap,occ],
                #            muteProgress=True
                #            ))**2 # dimension [eV^2]
                # ===============================================

                Q_co_point = kmap[Q_kmap, :3] # Q_kmap is the index in kmap.dat
                key_temp = '  %.5f    %.5f    %.5f' % (Q_co_point[0], Q_co_point[1], Q_co_point[2])
                Q_inteqp_index = Qq_dic[key_temp.replace('_','')]

                q_inteqp_point = grid_q_gqQ_res[q_qQmap]
                key_temp = '  %.5f    %.5f    %.5f' % (q_inteqp_point[0], q_inteqp_point[1], q_inteqp_point[2])
                q_inteqp_index = Qq_dic[key_temp.replace('_','')]

                Q_plus_q_inteqp_point = move_k_back_to_BZ_1(Q_co_point + q_inteqp_point)
                key_temp = '  %.5f    %.5f    %.5f' % (Q_plus_q_inteqp_point[0], Q_plus_q_inteqp_point[1], Q_plus_q_inteqp_point[2])
                Qpr_as_Q_plus_q_inteqp_index = Qq_dic[key_temp.replace('_','')]


                # Qpr_as_Q_plus_q_acv_index = Q_plus_q_kmapout[0]

                # tododone: double check this!!!


                # find energy for exciton and phonon (you should use index_acv and index_gkk)
                # omega.shape  = (nq, nmode)
                # exciton_energy.shape = (nQ, nS)
                # OMEGA_xx has included h_bar
                #==================================================================
                # gqQ_sq_inteqp.shape = (interpolate_size, interpolate_size)
                # res_omega.shape = (n_phonon, interpolate_size, interpolate_size)
                # res_OMEGA.shape = (n_exciton, interpolate_size, interpolate_size)

                #==================================================================

                g_nmvQ_temp = gqQ_sq_inteqp_temp[q_inteqp_index]
                # print("g_nmvQ_temp",g_nmvQ_temp)
                # omega_v_q_temp = res_omega[int(v_ph_gkk_index_loop)].flatten()[q_inteqp_index]
                # print("omega_q:", omega_v_q_temp)
                if g_nmvQ_temp == 0:
                    # print("g_nmvQ_temp == 0:", g_nmvQ_temp == 0)
                    continue

                omega_v_q_temp = res_omega[int(v_ph_gkk_index_loop)].flatten()[q_inteqp_index] * 10 ** (-3) # dimension [eV]
                OMEGA_n_Q_temp = res_OMEGA[int(n_ext_acv_index)].flatten()[Q_inteqp_index] # dimension [eV]
                OMEGA_m_Q_plus_q_temp = res_OMEGA[int(m_ext_acv_index_loop)].flatten()[Qpr_as_Q_plus_q_inteqp_index] # dimension [eV]

                # tododone Check this !!!
                # omega_v_q_temp     = omega_mat[int(q_gkk_index),int(v_ph_gkk_index_loop)] * 10 ** (-3) # dimension [eV]
                # OMEGA_m_Q_plus_q_temp = exciton_energy[int(Qpr_as_Q_plus_q_acv_index), int(m_ext_acv_index_loop)] # dimension [eV]
                # OMEGA_n_Q_temp        = exciton_energy[int(Q_acv_index),               int(m_ext_acv_index_loop)] # dimension [eV]

                distribution_first_temp = (
                                    (BE(omega=omega_v_q_temp, T=T) + 1 + BE(omega=OMEGA_m_Q_plus_q_temp, T=T))
                                     * Dirac_1(OMEGA_n_Q_temp - OMEGA_m_Q_plus_q_temp - omega_v_q_temp, sigma=degaussian))


                distribution_second_temp = (
                                    (BE(omega=omega_v_q_temp, T=T) - BE(omega=OMEGA_m_Q_plus_q_temp, T=T))
                                    * Dirac_1(OMEGA_n_Q_temp - OMEGA_m_Q_plus_q_temp + omega_v_q_temp, sigma=degaussian)

                                     )

                Gamma_first_res = Gamma_first_res + factor * g_nmvQ_temp * distribution_first_temp
                Gamma_second_res = Gamma_second_res + factor * g_nmvQ_temp * distribution_second_temp

                # print("distribution_first_temp", distribution_first_temp)
                # print("BE part:", (BE(omega=omega_v_q_temp, T=T) + 1 + BE(omega=OMEGA_m_Q_plus_q_temp, T=T)))
                # print("DIrac:", Dirac_1(OMEGA_n_Q_temp - OMEGA_m_Q_plus_q_temp - omega_v_q_temp, sigma=degaussian))
                # print("dirac energy",OMEGA_n_Q_temp - OMEGA_m_Q_plus_q_temp - omega_v_q_temp)
                # print("\n")
    # TODO: Discuss with Diana!!!

    # print(Gamma_first_res)
    # print(Gamma_second_res)

    if status_for_para == 'nopara':
        return Gamma_first_res + Gamma_second_res
    else:
        return Gamma_first_res + Gamma_second_res

#==============================================================================================================>>>>>>>

def interpolation_check_for_Gamma_calculation(interpo_size=4, path='./', mute=False):
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
        # print("n_fi:",n_fi)
        # print('n_co:',n_co)
        raise Exception("Only support integer multiple interpolation: k-grid after interpolation should cover k-grid before interpolation (e.g.: (4,4,1) --10--> (32, 32, 1))")
    else:
        if not mute:
            print("[interpolation size]: check")
    res0, new_q_out = gqQ_inteqp_get_coarse_grid(path=path, new_q_out=True)
    res_gqQ = gqQ_inteqp_q_series(res=res0,interpo_size=interpo_size,path=path,new_q_out=new_q_out)
    # Note: This is used for generating grid_q_gqQ_res, we will not use this interpolated gqQ here! 12/25/2022 Bowen Hou
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

#tododone: suggestion function for interpolation size
# rewrite Gamma Calculation, write document for kmap, k_dic, Qq_dic

# def Qpoints_2_Qfi_dic_generate(Q_grid, q_grid):
#     pass


if __name__ == "__main__":
    res0= Gamma_scat_test_nointeqp(Q_kmap=15, n_ext_acv_index=2,T=100, degaussian=0.001,path='../')
    # res = Gamma_scat_low_efficiency_inteqp(Q_kmap=15, n_ext_acv_index=2,T=100, degaussian=0.001,interposize=4,path='../')
    # res = interpolation_check_for_Gamma_calculation(path='../',interpo_size=12)




