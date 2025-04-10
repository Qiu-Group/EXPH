#!/usr/bin/env python

from IO.IO_get_parameter import read_kkqQ_number
from IO.IO_readinput import readin
from IO.IO_acv import create_acvsh5
from IO.IO_gkk import create_gkkh5
from Common.kgrid_check import k_grid_summary
from Common.band_check import band_summary
from Parallel.Para_EX_PH_mat import gqQ_h5_generator_Para
from Parallel.Para_EX_PH_scat import para_Gamma_scat_inteqp
from Parallel.Para_EX_PH_lifetime_all_Q import para_Exciton_Life_standard
from Parallel.Para_rt_Boltzmann_Diffusion_CPU import Solver_of_phase_space_CPU

from PLot_.plot_frame_evolution import plot_frame_diffusion
from PLot_.plot_xct_band import plot_exciton_band_inteqp
from PLot_.plot_phonon_band import plot_phonon_band_inteqp
from PLot_.plot_xct_lifetime import plot_ex_lifetime_inteqp
from PLot_.plot_xct_ph_mat import plot_ex_ph_mat_inteqp
from mpi4py import MPI
import sys
import time

# here is debug
# here debug is done

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

############################################# (1) read input ###########################################################

if rank == 0:
    start_time = time.time()
    start_time_proc = time.process_time()
    input = readin('./')
    [nQ, nq, nk, nmode] = read_kkqQ_number(input['save_path'])
    nband_gkk = int(input['conduction_end_gkk'] - input['valence_start_gkk'] + 1)
    sys.stdout.flush()
else:
    input = None
    [nQ, nq, nk, nmode] = [None, None, None, None]
    nband_gkk = None

# input = comm.scatter(input, root=0)
input = comm.bcast(input, root=0)
nQ = comm.bcast(nQ, root=0)
nq = comm.bcast(nq, root=0)
nk = comm.bcast(nk, root=0)
nmode = comm.bcast(nmode, root=0)
nband_gkk = comm.bcast(nband_gkk, root=0)

############################################ (2) Do Job ################################################################
# (i) initialization: creating Acv.h5, gkk.h5, kkqQmap.dat and bandmap.dat
if rank == 0:
    if 'initialize' in input and input['initialize'] == True: # initialize default is False, calculate it only when xct_scattering_rate is in input and set as True
        if 'S_list' not in input:
            raise Exception('S_list is not specified, it could be False pr [x,x]')
        elif input['S_list']:
            print('note: S_list: ', input['S_list'])
            input['S_list'] = [input['S_list'][0]-1, input['S_list'][-1]-1]
        else:
            print('note: S_list:', input['S_list'])
        create_acvsh5(nQ, save_path=input['save_path'] +  'acvs.save/', S_list=input['S_list'])
        sys.stdout.flush()
        create_gkkh5(nq,nk,nmode,nband_gkk, nband_gkk, save_path=input['save_path'] +  'gkk.save/')
        sys.stdout.flush()
        k_grid_summary(acvs_save_path=input['save_path'] +  'acvs.save/')
        sys.stdout.flush()
        band_summary(v_start_gkk=int(input['valence_start_gkk']),c_end_gkk=int(input['conduction_end_gkk']))
        sys.stdout.flush()
        init_status = "done" # synchronize all processor after init
    else:
        init_status = 'done' # synchronize all processor after init
else:
    init_status = 'todo' # synchronize all processor after init
    pass

#-----------------------------------------------------------------------------------------------------------------------
init_status = comm.bcast(init_status, root=0) # synchronize all processor after init

# (ii) Exciton-Phonon Matrix
if 'exph_mat_write' in input and input['exph_mat_write'] == True: # ex-ph_mat default is False, calculate it only when xct_scattering_rate is in input and set as True
    if rank == 0: # for print!
        print("\nStarting EX-PH matrix calculation:")
    if "nS_initial" in input and "nS_final" in input:
        if rank == 0: # for print!
            print("nS_initial: ", int(input["nS_initial"]))
            print("nS_final: ", int(input["nS_final"]))
            sys.stdout.flush()
        # place your job function here!
        # para_Gamma_scat_inteqp(Q_kmap=int(input["xct_scattering_rate_Q_kmap"] - 1),
        #                        n_ext_acv_index=int(input["xct_scattering_rate_nS"] - 1),
        #                        T=input["T"],
        #                        degaussian=input["degaussian"],
        #                        path=input["h5_path"],
        #                        interposize=int(input["xct_scattering_rate_interpolation"]),
        #                        muteProgress=True)
        gqQ_h5_generator_Para(nS_initial = int(input["nS_initial"]),
                              nS_final   = int(input["nS_final"]),
                              path=input["h5_path"],
                              mute=True
                              )
        sys.stdout.flush()
    else:
        raise Exception("key parameter missing for xct_scattering rate from input!")
else:
    pass


#-----------------------------------------------------------------------------------------------------------------------
init_status = comm.bcast(init_status, root=0) # synchronize all processor after init

# (iii) scattering rate
if 'xct_scattering_rate' in input and input['xct_scattering_rate'] == True: # xct_scattering_rate default is False, calculate it only when xct_scattering_rate is in input and set as True
    if rank == 0: # for print!
        print("\nStarting Calculating Scattering Rate:")
    if "xct_scattering_rate_Q_kmap" in input and "xct_scattering_rate_nS" in input and "xct_scattering_rate_interpolation" in input:
        if rank == 0: # for print!
            print("xct_scattering_rate_Q_kmap: ", int(input["xct_scattering_rate_Q_kmap"]))
            print("xct_scattering_rate_nS: ", int(input["xct_scattering_rate_nS"]))
            print("xct_scattering_rate_interpolation: ", int(input["xct_scattering_rate_interpolation"]))
            sys.stdout.flush()
        # place your job function here!
        para_Gamma_scat_inteqp(Q_kmap=int(input["xct_scattering_rate_Q_kmap"] - 1),
                               n_ext_acv_index=int(input["xct_scattering_rate_nS"] - 1),
                               T=input["T"],
                               degaussian=input["degaussian"],
                               path=input["h5_path"],
                               interposize=int(input["xct_scattering_rate_interpolation"]),
                               muteProgress=True)
        sys.stdout.flush()
    else:
        raise Exception("key parameter missing for xct_scattering rate from input!")
else:
    pass

#-----------------------------------------------------------------------------------------------------------------------
init_status = comm.bcast(init_status, root=0) # synchronize all processor after init

# (iv) lifetime
if 'xct_lifetime_all_BZ' in input and input['xct_lifetime_all_BZ'] == True: # xct_lifetime_all_BZ default is False, calculate it only when xct_lifetime_all_BZ is in input and set as True
    if rank == 0: # for print!
        print("\nStarting Calculating Exciton Life Time over 1st BZ:")
    if "xct_lifetime_all_BZ_nS" in input and "xct_lifetime_all_BZ_interpolation" in input:
        if rank == 0:
            print("xct_lifetime_all_BZ_nS: ",int(input["xct_lifetime_all_BZ_nS"]))
            print("xct_lifetime_all_BZ_interpolation", int(input["xct_lifetime_all_BZ_interpolation"]))
            sys.stdout.flush()

        para_Exciton_Life_standard(path=input["h5_path"],
                                   interposize=int(input["xct_lifetime_all_BZ_interpolation"]),
                                   T=input["T"],
                                   degaussian=input["degaussian"],
                                   write=True,
                                   n_ext_acv_index=int(input["xct_lifetime_all_BZ_nS"]-1))
        sys.stdout.flush()
    else:
        raise Exception("key parameter missing for xct_lifetime rate from input!")
else:
    pass


#-----------------------------------------------------------------------------------------------------------------------
init_status = comm.bcast(init_status, root=0) # synchronize all processor after init

# (v) Diffusion PDE
if 'Diffusion_PDE' in input and input['Diffusion_PDE'] == True: # xct_lifetime_all_BZ default is False, calculate it only when xct_lifetime_all_BZ is in input and set as True
    if rank == 0: # for print!
        print("\nStarting Solving Diffusion PDE:")
    if "delta_degaussian_occupation" in input \
            and "delta_T" in input\
            and "T_total" in input\
            and "delta_X" in input\
            and "delta_Y" in input\
            and "X" in input\
            and "Y" in input\
            and "initial_S_diffusion" in input\
            and "initial_Q_diffusion" in input\
            and "initial_Gaussian_Broad" in input:
        if rank == 0: # TODO: add onGPU
            # print("xct_lifetime_all_BZ_nS: ",int(input["xct_lifetime_all_BZ_nS"]))
            # print("xct_lifetime_all_BZ_interpolation", int(input["xct_lifetime_all_BZ_interpolation"]))
            print("delta_T: %s T_total: %s delta_X: %s delta_Y: %s X: %s Y: %s"
                  %(input['delta_T'] ,input['T_total'],  input['delta_X'], input['delta_Y'],input['X'],input['Y'] ))
            print("delta_degaussian_occupation: %s initial_S_diffusion: %s initial_Q_diffusion: %s initial_Gaussian_Broad: %s"
                  % (input["delta_degaussian_occupation"], input["initial_S_diffusion"], input['initial_Q_diffusion'], input['initial_Gaussian_Broad']))
            sys.stdout.flush()

        a = Solver_of_phase_space_CPU(degaussian=input["delta_degaussian_occupation"],
                                      delta_T=input['delta_T'],
                                      T_total=input['T_total'],
                                      T=input["T"],
                                      delta_X=input['delta_X'],
                                      delta_Y=input['delta_Y'],
                                      X=input['X'],
                                      Y=input['Y'],
                                      path=input["h5_path"],
                                      initial_S=int(input['initial_S_diffusion']-1),
                                      initial_Q=int(input['initial_Q_diffusion']-1),
                                      initial_Gaussian_Braod=input['initial_Gaussian_Broad'])
        a.solve_it()
        # a.write_diffusion_evolution()
        # para_Exciton_Life_standard(path=input["h5_path"],
        #                            interposize=int(input["xct_lifetime_all_BZ_interpolation"]),
        #                            T=input["T"],
        #                            degaussian=input["degaussian"],
        #                            write=True,
        #                            n_ext_acv_index=int(input["xct_lifetime_all_BZ_nS"]-1))
        sys.stdout.flush()
    else:
        raise Exception("key parameter missing for xct_lifetime rate from input!")
else:
    pass


############################################ (3) Plot ##################################################################
init_status = comm.bcast(init_status, root=0) # synchronize all processor after init
# (i) Plot Exciton Band

if rank == 0:
    if 'plot_xt_band' in input and input['plot_xt_band'] == True:
        if rank == 0:  # for print!
            print("\nPlotting Exciton Band:")
        if "plot_xt_band_number" in input and "plot_xt_interpolation" in input:
            if rank == 0:
                print("plot_xt_band_number: ", int(input["plot_xt_band_number"]))
                print("plot_xt_interpolation", int(input["plot_xt_interpolation"]))
                sys.stdout.flush()

            plot_exciton_band_inteqp(S_index=int(input["plot_xt_band_number"]-1),
                                     path=input["h5_path"],
                                     interposize=int(input["plot_xt_interpolation"]))
            sys.stdout.flush()
        else:
            raise Exception("key parameter missing for plotting exciton band!")
    else:
        pass
else:
    pass

#-----------------------------------------------------------------------------------------------------------------------
init_status = comm.bcast(init_status, root=0) # synchronize all processor after init
# (ii) Plot Phonon Band
if rank == 0:
    if 'plot_ph_band' in input and input['plot_ph_band'] == True:
        if rank == 0:  # for print!
            print("\nPlotting Exciton Band:")
        if "plot_ph_band_number" in input and "plot_ph_band_interpolation" in input:
            if rank == 0:
                print("plot_ph_band_number: ", int(input["plot_ph_band_number"]))
                print("plot_ph_band_interpolation", int(input["plot_ph_band_interpolation"]))
                sys.stdout.flush()

            plot_phonon_band_inteqp(V_index=int(input["plot_ph_band_number"]-1),
                                    path=input["h5_path"],
                                    interposize=int(input["plot_ph_band_interpolation"])
                                    )

            sys.stdout.flush()
        else:
            raise Exception("key parameter missing for plotting phonon band!")
    else:
        pass
else:
    pass

#-----------------------------------------------------------------------------------------------------------------------
init_status = comm.bcast(init_status, root=0) # synchronize all processor after init
# (iii) Plot Exciton Life Time
if rank == 0:
    if 'plot_xct_lifetime' in input and input['plot_xct_lifetime'] == True:
        if rank == 0:  # for print!
            print("\nPlotting Exciton Lifetime:")
        if 'plot_xct_lifetime_interposize_for_Lifetime' in input and "plot_xct_data" in input:
            if rank == 0:
                print("plot_xct_lifetime_interposize_for_Lifetime: ", int(input["plot_xct_lifetime_interposize_for_Lifetime"]))
                print("plot_xct_data: ", input["plot_xct_data"])
                sys.stdout.flush()

            plot_ex_lifetime_inteqp(path=input["h5_path"],
                                    read_file=input['plot_xct_data'],
                                    T=input["T"],
                                    degaussian=input["degaussian"],
                                    interposize_for_Lifetime=int(input["plot_xct_lifetime_interposize_for_Lifetime"]),
                                    start_from_zero=False)

            sys.stdout.flush()
        else:
            raise Exception("key parameter missing for plotting exciton lifetime!")
    else:
        pass
else:
    pass

#-----------------------------------------------------------------------------------------------------------------------
init_status = comm.bcast(init_status, root=0) # synchronize all processor after init
# (iv) Plot Exciton matrix
if rank == 0:
    if 'plot_xct_ph_mat' in input and input['plot_xct_ph_mat'] == True:
        if rank == 0:  # for print!
            print("\nPlotting Exciton-Phonon Matrix:")
        if 'plot_xct_ph_mat_Q' in input \
                and "plot_xct_ph_mat_n" in input \
                and 'plot_xct_ph_mat_m' in input \
                and "plot_xct_ph_mat_ph" in input \
                and "plot_xct_ph_mat_interpolation" in input:
            if rank == 0:
                print("plot_xct_ph_mat_Q: ", int(input["plot_xct_ph_mat_Q"]))
                print("plot_xct_ph_mat_n: ", int(input["plot_xct_ph_mat_n"]))
                print('plot_xct_ph_mat_m', input['plot_xct_ph_mat_m'])
                print("plot_xct_ph_mat_ph", input['plot_xct_ph_mat_ph'])
                print("plot_xct_ph_mat_interpolation", int(input["plot_xct_ph_mat_interpolation"]))

                sys.stdout.flush()

            plot_ex_ph_mat_inteqp(Q_kmap_star=int(input["plot_xct_ph_mat_Q"] - 1),
                                  n_ex_acv=int(input["plot_xct_ph_mat_n"] - 1),
                                  m_ex_acv=[x-1 for x in input['plot_xct_ph_mat_m']],
                                  v_ph_gkk=[x-1 for x in input['plot_xct_ph_mat_ph']],
                                  mute=False,
                                  path=input["h5_path"],
                                  interposize=int(input["plot_xct_ph_mat_interpolation"]))

            sys.stdout.flush()
        else:
            raise Exception("key parameter missing for plotting exciton lifetime!")
    else:
        pass
else:
    pass



#-----------------------------------------------------------------------------------------------------------------------
init_status = comm.bcast(init_status, root=0) # synchronize all processor after init
# (v) Plot Diffusion
# print(input)
if rank == 0:
    if 'plot_diffusion' in input and input['plot_diffusion'] == True:
        if rank == 0:  # for print!
            print("\nPlotting Diffusion Frame:")
        if "plot_diffusion_state" in input \
                and "plot_diffusion_frame" in input\
                and 'Q1' in input and 'Q2' in input and 'Q3' in input and 'Q4' in input:
            if rank == 0:
                print("plot_diffusion_state: ", int(input["plot_diffusion_state"]))
                print("plot_diffusion_frame", int(input["plot_diffusion_frame"]))
                print("Q1: %s, Q2: %s, Q3: %s, Q4: %s"%(int(input["Q1"]), int(input["Q2"]),int(input["Q3"]),int(input["Q4"]))  )
                sys.stdout.flush()

            plot_frame_diffusion(i=int(input["plot_diffusion_frame"])-1,
                                 path=input["h5_path"],
                                 S = int(input["plot_diffusion_state"]) - 1,
                                 Q1=int(input["Q1"])-1, Q2=int(input["Q2"])-1,
                                 Q3=int(input["Q3"])-1, Q4=int(input["Q4"])-1)

            sys.stdout.flush()
        else:
            raise Exception("key parameter missing for plotting phonon band!")
    else:
        pass
else:
    pass

if rank == 0:
    print("\nJob Done!")
    print("----------Time Summary----------")
    end_time = time.time()
    end_time_proc = time.process_time()
    print("the wall time is: %.3f s      |" % (end_time - start_time))
    print("the proc time is: %.3f s      |" % (end_time_proc - start_time_proc))
    print("--------------------------------")

    # hi, this is merged from parallel_high_energy