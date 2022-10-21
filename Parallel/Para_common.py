import numpy as np


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
