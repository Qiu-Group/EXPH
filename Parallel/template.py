from mpi4py import MPI
import time
from time import process_time
from mpi4py import MPI
from Common.progress import ProgressBar
from Parallel.Para_common import plan_maker, before_parallel_job, after_parallel_sum_job

# ========================def: input===================

# ===================initialization=====================
# 1.0 initialization:
# (1) create comm
# (2) set plan_list as None for every one
# (3) load data for every proc tododone: summarize to a template or function
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
plan_list = None

# your code is here (modify it):
workload_over_kmap = 'something'
# your code is here (some other input needed to be initialize)
#...

#-----------------------------------------------------

# ========================plan=========================
# 2.0 Plan before calculation
plan_list, start_time, start_time_proc = before_parallel_job(rk=rank, size=size, workload_para=workload_over_kmap)
# (b) distribute plan
plan_list = comm.scatter(plan_list, root=0)
print('process_%d. plan is ' % rank, plan_list, 'workload:', plan_list[-1] - plan_list[0])

#======================calculation=====================
# 3.0 calculation
# (a) each proc is doing job!
# (b) progress

# your code is here (modify it):
res = "job_func(kwarg)" # job_func needed to be divided
res_rcev_to_0 = comm.gather('res for each job',root=0)

# ======================collection=====================
value = after_parallel_sum_job(rk=rank, size=size, receive_res=res_rcev_to_0, start_time=start_time,
                           start_time_proc=start_time_proc)
# if rank == 0:
#     return value
