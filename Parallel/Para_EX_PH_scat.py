from ELPH.EX_PH_scat import Gamma_scat_low_efficiency_inteqp
import numpy as np
from Common.distribution import BE, FD, Dirac_1, Dirac_2
from IO.IO_gkk import read_omega, read_gkk
from IO.IO_acv import read_Acv, read_Acv_exciton_energy
from IO.IO_common import read_bandmap, read_kmap, construct_kmap
from ELPH.EX_PH_mat import gqQ, gqQ_inteqp_q_series
from Common.progress import ProgressBar
from Common.common import move_k_back_to_BZ_1, equivalence_order
from ELPH.EX_PH_inteqp import omega_inteqp_q,OMEGA_inteqp_Q
from ELPH.EX_PH_scat import interpolation_check_for_Gamma_calculation
from mpi4py import MPI
from ELPH.EX_PH_mat import gqQ_inteqp_get_coarse_grid, gqQ_inteqp_q_series
from Parallel.Para_common import before_parallel_job, after_parallel_sum_job

# (1) para_Gamma_scat_low_efficiency_inteqp: it could calculate Gamma scat, but efficiency is pretty low! It is not good for parallel

# --> (2) para_Gamma_scat_inteqp: it is used for parallel!



def para_Gamma_scat_low_efficiency_inteqp(Q_kmap=15, n_ext_acv_index=2,T=100, degaussian=0.001 , interposize=4, path='./'):
    # input===================
    # Q_kmap=15
    # n_ext_acv_index=2
    # T=100
    # degaussian=0.001
    # path='../'


    # ===================initialization=====================
    # 1.0 initialization:
    # (1) create comm
    # (2) set plan_list as None for every one
    # (3) load data for every proc tododone: summarize to a template or function
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    plan_list = None

    # workload_over_qmap = len(read_kmap(path=path))
    workload_over_qQmap = interposize**2

    # 2.0 Plan before calculation
    plan_list, start_time, start_time_proc = before_parallel_job(rk=rank, size=size, workload_para=workload_over_qQmap)
    # (b) distribute plan
    plan_list = comm.scatter(plan_list, root=0)
    print('process_%d. plan is ' % rank, plan_list, 'workload:', plan_list[-1]-plan_list[0])

    # ======================calculation=====================
    # 3.0 calculation
    # (a) each proc is doing job!
    # (b) progress
    # print(type(plan_list[0]))
    # print(plan_list)
    # print(plan_list[0],plan_list[-1])
    # [res_first_each_proc, res_second_each_proc, factor_first_each_proc, factor_second_each_proc]= Gamma_scat(Q_kmap=Q_kmap,
    #                            n_ext_acv_index=n_ext_acv_index,
    #                            T=T,
    #                            degaussian=degaussian,
    #                            muteProgress=True,
    #                            path=path,
    #                            q_map_start_para=plan_list[0],
    #                            q_map_end_para=plan_list[-1])

    #Warning: since we need to add all normalization after finishing all loops, so here we need pass them separately, and assemble them together after finishi all
    res_each_proc = Gamma_scat_low_efficiency_inteqp(Q_kmap=Q_kmap,
                               n_ext_acv_index=n_ext_acv_index,
                               T=T,
                               degaussian=degaussian,
                               interposize=interposize,
                               muteProgress=True,
                               path=path,
                               q_map_start_para=plan_list[0],
                               q_map_end_para=plan_list[-1])
    print(res_each_proc)
    # res_first_rcev_to_0 = comm.gather(res_first_each_proc, root=0)
    # res_second_rcev_to_0 = comm.gather(res_second_each_proc, root=0)
    # factor_first_rcev_to_0 = comm.gather(factor_first_each_proc, root=0)
    # factor_second_rcev_to_0 = comm.gather(factor_second_each_proc, root=0)
    res_rcev_to_0 = comm.gather(res_each_proc, root=0)

    # ======================collection=====================
    # value_first = after_parallel_sum_job(rk=rank, size=size, receive_res=res_first_rcev_to_0 , start_time=start_time,
    #                            start_time_proc=start_time_proc,mute=True)
    # value_second = after_parallel_sum_job(rk=rank, size=size, receive_res=res_second_rcev_to_0 , start_time=start_time,
    #                            start_time_proc=start_time_proc,mute=True)
    # factor_first = after_parallel_sum_job(rk=rank, size=size, receive_res=factor_first_rcev_to_0 , start_time=start_time,
    #                            start_time_proc=start_time_proc,mute=True)
    # factor_second = after_parallel_sum_job(rk=rank, size=size, receive_res=factor_second_rcev_to_0 , start_time=start_time,
    #                            start_time_proc=start_time_proc,mute=True)
    value = after_parallel_sum_job(rk=rank, size=size, receive_res=res_rcev_to_0 , start_time=start_time,
                               start_time_proc=start_time_proc,mute=False)
    if rank==0:
        # print('===================================')
        # print('process= %d is summarizing ' % rank)
        # value = value_first/factor_first + value_second/factor_second
        # # value =value_first + value_second
        # print("res is", value)
        # end_time = time.time()
        # end_time_proc = process_time()
        # print("the wall time is: %.3f s" % (end_time - start_time))
        # print("the proc time is: %.3f s" % (end_time_proc - start_time_proc))
        # print('===================================')
        # print('hello')

        return value


def para_Gamma_scat_inteqp(Q_kmap=15, n_ext_acv_index=2,T=100, degaussian=0.001, interposize=4, interpolation_check_res = None,
               muteProgress=True, path='./'):
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

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    plan_list = None
    plan_list_2 = None

    if not interpolation_check_res:
        [grid_q_gqQ_res, Qq_dic, res_omega, res_OMEGA] = interpolation_check_for_Gamma_calculation(interpo_size=interposize,path=path,mute=muteProgress)
    else:
        [grid_q_gqQ_res, Qq_dic, res_omega, res_OMEGA] = interpolation_check_res
        interposize = int(np.sqrt(interpolation_check_res[0].shape[0]))


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
    Nqpt = interposize**2 # Nice!

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


    workload_over_q_co = len(kmap)
    workload_over_q_fi = interposize**2

    plan_list, start_time, start_time_proc = before_parallel_job(rk=rank, size=size, workload_para=workload_over_q_co)
    plan_list = comm.scatter(plan_list, root=0)
    print('process_%d. plan is ' % rank, plan_list, 'workload:', plan_list[-1] - plan_list[0])
    plan_list_2, start_time_2, start_time_proc_2 = before_parallel_job(rk=rank, size=size, workload_para=workload_over_q_fi)
    plan_list_2 = comm.scatter(plan_list_2, root=0)
    # print('process_%d. plan is ' % rank, plan_list, 'workload:', plan_list[-1]-plan_list[0])




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
            # gqQ_sq_inteqp_temp = np.abs(gqQ_inteqp_q(n_ex_acv_index=n_ext_acv_index,
            #                                   m_ex_acv_index=m_ext_acv_index_loop,
            #                                   v_ph_gkk=v_ph_gkk_index_loop,
            #                                   Q_kmap=Q_kmap, #!!! this Q_kmap from function parameter
            #                                   interpo_size=interposize,
            #                                   new_q_out=False,
            #                                   acvmat=acvmat,
            #                                   gkkmat=gkkmat,
            #                                   kmap=kmap,
            #                                   kmap_dic=kmap_dic,
            #                                   bandmap_occ= [bandmap,occ],
            #                                   muteProgress=True,
            #                                   ))**2 # unit [eV^2]

            res_temp_each_process, new_q_out = gqQ_inteqp_get_coarse_grid(n_ex_acv_index=n_ext_acv_index,
                                                                          m_ex_acv_index=m_ext_acv_index_loop,
                                                                          v_ph_gkk=v_ph_gkk_index_loop,
                                                                          Q_kmap=Q_kmap, #interpo_size=12
                                                                          new_q_out=False,
                                                                          acvmat=acvmat,
                                                                          gkkmat=gkkmat,
                                                                          kmap=kmap,
                                                                          kmap_dic=kmap_dic,
                                                                          bandmap_occ=[bandmap,occ],
                                                                          muteProgress=True,
                                                                          path=path,
                                                                          q_map_start_para=plan_list[0],
                                                                          q_map_end_para=plan_list[1])

            res_rcev_to_0 = comm.gather(res_temp_each_process, root=0)
            gqQ_sq_inteqp_temp_co = after_parallel_sum_job(rk=rank, size=size, receive_res=res_rcev_to_0, start_time=start_time,
                                           start_time_proc=start_time_proc,mute=True)
            if n_ext_acv_index==2 and m_ext_acv_index_loop==0 and v_ph_gkk_index_loop ==3 and Q_kmap==15:
                # print("gqQ_sq_temp:",gqQ_sq_inteqp_temp_co)
                pass
            # gqQ_sq_inteqp.shape = (interpolate_size, interpolate_size)
            # gqQ_sq_inteqp.flatten.shape = (interpolate_size**2, 1)
            # same order as grid_q_gqQ_res
            if rank == 0:
                gqQ_sq_inteqp_temp_fi = gqQ_inteqp_q_series(res=gqQ_sq_inteqp_temp_co, new_q_out=new_q_out, path=path, interpo_size=interposize)
                #
                # if n_ext_acv_index == 2 and m_ext_acv_index_loop == 0 and v_ph_gkk_index_loop == 3 and Q_kmap == 15:
                #     print("gqQ_sq_temp:",gqQ_sq_inteqp_temp_fi[0])
                 #   pass
                gqQ_sq_inteqp = (gqQ_sq_inteqp_temp_fi**2).flatten()
            else:
                gqQ_sq_inteqp = 0


            gqQ_sq_inteqp = comm.bcast(gqQ_sq_inteqp,root=0)





            for q_qQmap in range(plan_list_2[0],plan_list_2[1]):  # q_kmap is the index of kmap from point 0-15 in kmap.dat (e.g)


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

                g_nmvQ_temp = gqQ_sq_inteqp[q_inteqp_index]
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

                Gamma_res = Gamma_res + (factor * g_nmvQ_temp * distribution_first_temp + factor * g_nmvQ_temp * distribution_second_temp)
                # Gamma_second_res = Gamma_second_res +

                # print("distribution_first_temp", distribution_first_temp)
                # print("BE part:", (BE(omega=omega_v_q_temp, T=T) + 1 + BE(omega=OMEGA_m_Q_plus_q_temp, T=T)))
                # print("DIrac:", Dirac_1(OMEGA_n_Q_temp - OMEGA_m_Q_plus_q_temp - omega_v_q_temp, sigma=degaussian))
                # print("dirac energy",OMEGA_n_Q_temp - OMEGA_m_Q_plus_q_temp - omega_v_q_temp)
                # print("\n")

    # Gamma_first_to_0 = comm.gather(Gamma_first_res, root=0)
    # Gamma_second_to_0 = comm.gather(Gamma_second_res, root=0)
    # Gamma_fisrt_val  = after_parallel_sum_job(rk=rank, size=size, receive_res=Gamma_first_to_0 , start_time=start_time,
    #                            start_time_proc=start_time_proc,mute=True)
    # Gamma_second_val = after_parallel_sum_job(rk=rank, size=size, receive_res=Gamma_second_to_0 , start_time=start_time,
    #                            start_time_proc=start_time_proc,mute=True)

    Gamma_res_to_0 = comm.gather(Gamma_res, root=0)
    Gamma_res_val =    after_parallel_sum_job(rk=rank, size=size, receive_res=Gamma_res_to_0 , start_time=start_time,
                               start_time_proc=start_time_proc,mute=False)

    # TODO: Discuss with Diana!!!
    # print("final value is",Gamma_fisrt_val)
    # print("final value is",Gamma_second_val)
    if rank == 0:
        return Gamma_res_val

#==============================================================================================================>>>>>>>



# tododone: find a parallel_over_sum!!! use this as a test suite
# tododone: intensive test needed to be done after lunch
# tododone: para_fun(job_fun, *kwarg): use *kwarg to pass parameters to job_func
# TODO: test parallel efficiency for interpolation!!
#  the worst part of parallel is from 321-333 lines EX_PH_scat.py,
#  which leands to a non-Linear parallel, since use interpolation for many times.
if __name__ == "__main__":
    # res_para = para_Gamma_scat_low_efficiency_inteqp(Q_kmap=15, n_ext_acv_index=2,T=100, degaussian=0.001, interposize=4, path='../')
    res = para_Gamma_scat_inteqp(Q_kmap=15, n_ext_acv_index=2, T=100, degaussian=0.001,path='../',interposize=4)
    pass