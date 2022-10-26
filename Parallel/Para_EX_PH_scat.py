from ELPH.EX_PH_scat import Gamma_scat
from Parallel.Para_common import plan_maker, before_parallel_job, after_parallel_sum_job
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

    # Warning: since we need to add all normalization after finishing all loops, so here we need pass them separately, and assemble them together after finishi all
    res_each_proc = Gamma_scat(Q_kmap=Q_kmap,
                               n_ext_acv_index=n_ext_acv_index,
                               T=T,
                               degaussian=degaussian,
                               muteProgress=True,
                               path=path,
                               q_map_start_para=plan_list[0],
                               q_map_end_para=plan_list[-1])

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


# todo: find a parallel_over_sum!!! use this as a test suite
# todo: intensive test needed to be done after lunch
# todo: para_fun(job_fun, *kwarg): use *kwarg to pass parameters to job_func
if __name__ == "__main__":
    res_para = para_Gamma_scat(Q_kmap=15, n_ext_acv_index=2,T=100, degaussian=3, path='../')