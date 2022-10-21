from ELPH.EX_Ph_scat import Gamma_scat
from Parallel.Para_common import plan_maker, before_parallel_job, after_parallel_job
from mpi4py import MPI
from IO.IO_common import construct_kmap, read_kmap, read_bandmap, readkkqQ
from IO.IO_gkk import read_gkk
from IO.IO_acv import read_Acv
from time import  process_time
import time


def para_Gamma_scat(Q_kmap=15, n_ext_acv_index=2,T=100, degaussian=0.001, path='./'):
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

    workload_over_qmap = len(read_kmap(path=path))

    # 2.0 Plan before calculation
    plan_list, start_time, start_time_proc = before_parallel_job(rk=rank, size=size, workload_para=workload_over_qmap)
    # (b) distribute plan
    plan_list = comm.scatter(plan_list, root=0)
    print('process_%d. plan is ' % rank, plan_list)

    # ======================calculation=====================
    # 3.0 calculation
    # (a) each proc is doing job!
    # (b) progress
    res_each_proc = Gamma_scat(Q_kmap=Q_kmap,
                               n_ext_acv_index=n_ext_acv_index,
                               T=T,
                               degaussian=degaussian,
                               muteProgress=True,
                               path=path,
                               q_map_start_para=plan_list[0],
                               q_map_end_para=plan_list[-1])

    res_rcev_to_0 = comm.gather(res_each_proc, root=0)


    # ======================collection=====================
    value = after_parallel_job(rk=rank, size=size, receive_res=res_rcev_to_0, start_time=start_time,
                               start_time_proc=start_time_proc)
    if rank==0:
        return value


# todo: find a parallel_over_sum!!! use this as a testsuite
# todo: intensive test needed to be done after lunch
# todo: para_fun(job_fun, *kwarg): use *kwarg to pass parameters to job_func
if __name__ == "__main__":
    res_para = para_Gamma_scat(Q_kmap=15, n_ext_acv_index=2,T=100, degaussian=0.001, path='../')