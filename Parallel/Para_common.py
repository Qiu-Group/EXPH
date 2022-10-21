import numpy as np
import time
from time import process_time

def plan_maker(nproc, nloop):
    """
    this function help proc_0 make a plan for every other processor
    :param nproc: number of processor
    :param nloop:
    :return:
        """
    # nloop = 20
    # nproc = 15

    workload_per_proc = nloop // nproc
    workload_per_proc_array = np.ones(nproc) * workload_per_proc
    workload_left = nloop%nproc
    plan_list = []

    if workload_left != 0:
        print('warning: proc is noe divided evenly')

    for i in range(workload_left):
        workload_per_proc_array[i] = workload_per_proc_array[i] + 1

    pointer_begin = 0


    for j in range(nproc):

        plan_list.append([int(pointer_begin), int(pointer_begin + workload_per_proc_array[j])])
        pointer_begin = pointer_begin + workload_per_proc_array[j]

    return plan_list

def before_parallel_job(rk,size, workload_para):
    """
    :param rk: rank of job
    :return:
    """
    # todo: not finish: problem: for process which is not 0, how to return plan_list, time
    if rk == 0:
        # status = 'start'
        start_time = time.time()
        start_time_proc = process_time()

        plan_list = plan_maker(nproc=size, nloop=workload_para)
        print('process_%d finish plan:' % rk, plan_list)
        return plan_list, start_time, start_time_proc
    else:
        return None, None, None

def after_parallel_job(rk,size,receive_res,start_time,start_time_proc):
    if rk == 0:
        value = 0
        print('===================================')
        print('process= %d is summarizing ' % rk)
        for k in range(size):
            value = value + receive_res[k]
        print("sub_res is", receive_res)
        print("res is",value)
        end_time = time.time()
        end_time_proc = process_time()
        print("the wall time is: %.3f s"%(end_time - start_time))
        print("the proc time is: %.3f s"%(end_time_proc - start_time_proc))
        print('===================================')
        print('hello')
        return value
    else:
        pass