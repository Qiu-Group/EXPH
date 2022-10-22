import numpy as np

from ELPH.EX_PH_lifetime_all_Q import Exciton_Life
from Parallel.Para_common import plan_maker, before_parallel_job, after_parallel_sum_job
from mpi4py import MPI
from IO.IO_common import construct_kmap, read_kmap, read_bandmap, readkkqQ
from IO.IO_gkk import read_gkk
from IO.IO_acv import read_Acv
from time import  process_time
import time

n_ext_acv_index=0
T=100
degaussian = 0.001
path='../'
Q_kmap_start_para='nopara'
Q_kmap_end_para='nopara'
outfilename = 'ex_S%s_lifetime.dat'%(n_ext_acv_index + 1)

# ===================initialization=====================
# 1.0 initialization:
# (1) create comm
# (2) set plan_list as None for every one
# (3) load data for every proc tododone: summarize to a template or function
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
plan_list = None
# parallel over Q_kmap
workload_over_Qmap = len(read_kmap(path=path))
#-----------------------------------------------------


# ========================plan=========================
# 2.0 Plan before calculation
plan_list, start_time, start_time_proc = before_parallel_job(rk=rank, size=size, workload_para=workload_over_Qmap)
# (b) distribute plan
plan_list = comm.scatter(plan_list, root=0)
print('process_%d. plan is ' % rank, plan_list, 'workload:', plan_list[-1] - plan_list[0])

# ======================calculation=====================
# 3.0 calculation
# (a) each proc is doing job!
# (b) progress
# res_each_proc = Gamma_scat(Q_kmap=Q_kmap,
#                            n_ext_acv_index=n_ext_acv_index,
#                            T=T,
#                            degaussian=degaussian,
#                            muteProgress=True,
#                            path=path,
#                            q_map_start_para=plan_list[0],
#                            q_map_end_para=plan_list[-1])

res_each_proc = Exciton_Life(n_ext_acv_index=n_ext_acv_index,
                             T=T,
                             degaussian=degaussian,
                             path=path,
                             Q_kmap_start_para=plan_list[0],
                             Q_kmap_end_para=plan_list[-1],
                             mute=True)

res_rcev_to_0 = comm.gather(res_each_proc, root=0)

# ======================collection=====================
value = after_parallel_sum_job(rk=rank, size=size, receive_res=res_rcev_to_0, start_time=start_time,
                               start_time_proc=start_time_proc)
# write down value
if rank == 0:
    np.savetxt(outfilename,value)