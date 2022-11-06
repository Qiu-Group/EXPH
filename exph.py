from IO.IO_get_parameter import read_kkqQ_number
from IO.IO_readinput import readin
from IO.IO_acv import create_acvsh5
from IO.IO_gkk import create_gkkh5
from Common.kgrid_check import k_grid_summary
from Common.band_check import band_summary
from Parallel.Para_EX_PH_scat import para_Gamma_scat_inteqp
from Parallel.Para_EX_PH_lifetime_all_Q import para_Exciton_Life_standard
from mpi4py import MPI
import sys
import time


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
        create_acvsh5(nQ, save_path=input['save_path'] +  'acvs.save/')
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

# (ii) scattering rate
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

# (iii) lifetime
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
        raise Exception("key parameter missing for xct_scattering rate from input!")
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
