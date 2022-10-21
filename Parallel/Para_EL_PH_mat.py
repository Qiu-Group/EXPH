from ELPH.EL_PH_mat import gqQ
from Parallel.Para_common import plan_maker
from mpi4py import MPI
from IO.IO_common import construct_kmap, read_kmap, read_bandmap, readkkqQ
from IO.IO_gkk import read_gkk
from IO.IO_acv import read_Acv
from time import  process_time
import time

#def func:------input to be passed to function

def para_gqQ(n_ex_acv_index=8, m_ex_acv_index=3, v_ph_gkk=2, Q_kmap=3, q_kmap=11, acvmat=None, gkkmat=None,
            kmap=None, kmap_dic=None, bandmap_occ=None,  path='./'):

    # n_ex_acv_index=8
    # m_ex_acv_index=3
    # v_ph_gkk=2
    # Q_kmap=3
    # q_kmap=11
    # path='../'
    if acvmat is None:
        acvmat = read_Acv(path=path+'Acv.h5')
    if gkkmat is None:
        gkkmat = read_gkk(path=path+'gkk.h5')
    if kmap is None:
        kmap = read_kmap(path=path+'kkqQmap.dat')
    if kmap_dic is None:
        kmap_dic = construct_kmap(path=path+'kkqQmap.dat')
    if bandmap_occ is None:
        bandmap_occ = read_bandmap(path=path+'bandmap.dat')

    # ===================initialization=====================
    # 1.0 initialization:
    # (1) create comm
    # (2) set plan_list as None for every one
    # (3) load data for every proc todo: summarize to a template or function
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    plan_list = None
    workload_over_kmap = len(read_kmap(path=path+'kkqQmap.dat'))
    # -----------------------------------------------------

    # ========================plan=========================
    # 2.0 Plan before calculation
    # (a) proc_0 makes plan for all procs (including proc_0)
    if rank == 0:
        # status = 'start'
        start_time = time.time()
        start_time_proc = process_time()

        plan_list = plan_maker(nproc=size, nloop=workload_over_kmap)
        print('process_%d finish plan:'%rank, plan_list)
    else:
        pass
    # (b) distribute plan
    plan_list = comm.scatter(plan_list,root=0)
    print('process_%d. plan is ' % rank, plan_list)
    # print('process= %d. plan is '%rank, plan_list)



    #======================calculation=====================
    # 3.0 calculation
    # (a) each proc is doing job!
    # (b) progress
    res_each_proc=gqQ(n_ex_acv_index=n_ex_acv_index,
                      m_ex_acv_index=m_ex_acv_index,
                      v_ph_gkk=v_ph_gkk,
                      Q_kmap=Q_kmap,
                      q_kmap=q_kmap,
                      path=path,
                      acvmat=acvmat,
                      gkkmat=gkkmat,
                      kmap=kmap,
                      kmap_dic=kmap_dic,
                      bandmap_occ=bandmap_occ,
                      k_map_start_para=plan_list[0],
                      k_map_end_para=plan_list[-1],
                      muteProgress=True)

    res_rcev_to_0 = comm.gather(res_each_proc,root=0)

    if rank == 0:
        value = 0
        print('===================================')
        print('process_%d is summarizing ' % rank)
        for k in range(size):
            value = value + res_rcev_to_0[k]
        print("sub_res is", res_rcev_to_0)
        print("res is", value)
        end_time = time.time()
        end_time_proc = process_time()
        print("the wall time is: %.3f s" % (end_time - start_time))
        print("the proc time is: %.3f s" % (end_time_proc - start_time_proc))
        print('===================================')
        print('para version')
        return value

# todo: find a parallel_over_sum!!! use this as a testsuite
# todo: intensive test needed to be done after lunch
# todo: para_fun(job_fun, *kwarg): use *kwarg to pass parameters to job_func
res_para = para_gqQ(path='../')