from ELPH.EX_PH_mat import gqQ
from Parallel.Para_common import plan_maker, before_parallel_job, after_parallel_sum_job
from mpi4py import MPI
from IO.IO_common import construct_kmap, read_kmap, read_bandmap, readkkqQ
from IO.IO_gkk import read_gkk
from IO.IO_acv import read_Acv
from time import  process_time
import time
from IO.IO_common import read_kmap, read_lattice
from Common.common import frac2carte
from Common.progress import ProgressBar
import numpy as np
import h5py as h5
from IO.IO_gkk import read_omega, read_gkk
from IO.IO_acv import read_Acv, read_Acv_exciton_energy
import sys

#def func:------input to be passed to function

def para_gqQ(n_ex_acv_index=8, m_ex_acv_index=3, v_ph_gkk=2, Q_kmap=3, q_kmap=11,
             acvmat=None, gkkmat=None,kmap=None, kmap_dic=None, bandmap_occ=None,  path='./',mute=False):
    """
    !!! parallel over k_map !!!
    !!! PARALLEL OVER K_MAP !!!
    This function construct gnmv(Q,q)
    :param n_ex_acv: index of initial exciton state
    :param m_ex_acv: index of final exciton state
    :param v_ph_gkk: index of phonon mode
    :param Q_kmap: exciton momentum in kmap
    :param q_kmap: phonon momentumB in kmap
    :param acvmat: acv matrix (do not read it every time): False -> no input, read it
    :param gkkmat: gkk matrix (do not read it every time):  False -> no input, read it
    :param kmap: kmap matrix (do not read it every time) -> kmap.shape = (kx,ky,kz,Q, k_acv, q, k_gkk):  False -> no input, read it
    :param kmap_dic: kmap dictionary -> kmap_dic = {'  %.5f    %.5f    %.5f' : [Q, k_acv, q, k_gkk], ...}:  False -> no input, read it
    :param bandmap_occ: [bandmap_matrix, occ]:  False -> no input, read it
    :param path: path of *h5 and *dat
    :return: the gkk unit is meV, but return here is eV
    """
    # n_ex_acv_index=8
    # m_ex_acv_index=3
    # v_ph_gkk=2
    # Q_kmap=3
    # q_kmap=11
    # path='../'

    # ===================initialization=====================
    # 1.0 initialization:
    # (1) create comm
    # (2) set plan_list as None for every one
    # (3) load data for every proc todo: summarize to a template or function
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    plan_list = None

    # -----------------------------------------------------

    if acvmat is None:
        acvmat = read_Acv(path=path)
    if gkkmat is None:
        gkkmat = read_gkk(path=path)
    if kmap is None:
        kmap = read_kmap(path=path)
    if kmap_dic is None:
        kmap_dic = construct_kmap(path=path)
    if bandmap_occ is None:
        bandmap_occ = read_bandmap(path=path)

    workload_over_kmap = len(kmap)

    # ========================plan=========================
    # 2.0 Plan before calculation
    # (a) proc_0 makes plan for all procs (including proc_0)
    # if rank == 0:
    #     # status = 'start'
    #     start_time = time.time()
    #     start_time_proc = process_time()
    #
    #     plan_list = plan_maker(nproc=size, nloop=workload_over_kmap)
    #     print('process_%d finish plan:'%rank, plan_list)
    # else:
    #     pass

    plan_list, start_time, start_time_proc = before_parallel_job(rk=rank,size=size,workload_para=workload_over_kmap,mute=mute)

    # (b) distribute plan
    plan_list = comm.scatter(plan_list,root=0)
    if not mute:
        print('process_%d. plan is ' % rank, plan_list, 'workload:', plan_list[-1]-plan_list[0])
    # print('process= %d. plan is '%rank, plan_list)



    #======================calculation=====================
    # 3.0 calculation
    # (a) each proc is doing job!
    # (b) progress
    t_s = time.time()
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
    t_e1 =time.time()
    res_rcev_to_0 = comm.gather(res_each_proc,root=0)
    t_e2 = time.time()

    if not mute:
        print('te1 - ts:', t_e1-t_s)
        print("te2 - ts:", t_e2-t_s)
    # ======================collection=====================
    # if rank == 0:
    #     value = 0
    #     print('===================================')
    #     print('process_%d is summarizing ' % rank)
    #     for k in range(size):
    #         value = value + res_rcev_to_0[k]
    #     print("sub_res is", res_rcev_to_0)
    #     print("res is", value)
    #     end_time = time.time()
    #     end_time_proc = process_time()
    #     print("the wall time is: %.3f s" % (end_time - start_time))
    #     print("the proc time is: %.3f s" % (end_time_proc - start_time_proc))
    #     print('===================================')
    #     print('para version')
    #     return value
    value = after_parallel_sum_job(rk=rank, size=size, receive_res=res_rcev_to_0, start_time=start_time,
                               start_time_proc=start_time_proc,mute=mute)
    if rank==0:
        return value

# tododone: find a parallel_over_sum!!! use this as a testsuite
# tododone: intensive test needed to be done after lunch
# tododone: para_fun(job_fun, *kwarg): use *kwarg to pass parameters to job_func

def gqQ_h5_generator_Para(nS_initial = 0, nS_final = 0, path='./',mute=True):
    """
    Parallel efficiency is very high!! It is over nQ*nq
    :param Q_kmap_star:
    :param n_ex_acv:
    :param m_ex_acv:
    :param v_ph_gkk:
    :param path:
    :param mute:
    :return: |G(Q_kmap,q_kmap,n,m,v)|**2: exciton-phonon matrix
    All inputs are consistent with gqQ
    Note: Q_kmap/q_kmap is index of kmap (0-->nk). If you want to find Q=10, please find corresponding Q_kmap (0-nQ), then find this in gqQ
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # (a) load necessary information
    bvec = read_lattice('b',path)
    acvmat = read_Acv(path=path)
    gkkmat = read_gkk(path=path)
    kmap = read_kmap(path=path)
    kmap_dic = construct_kmap(path=path)
    bandmap_occ = read_bandmap(path=path)
    exciton_energy = read_Acv_exciton_energy(path=path)
    omega_mat = read_omega(path=path) # dimension [meV]
    n_phonon = omega_mat.shape[1]

    if nS_final == 0:
        nS_final = exciton_energy.shape[1]
    if nS_initial == 0:
        nS_initial = exciton_energy.shape[1]
    if nS_final > exciton_energy.shape[1] or nS_initial > exciton_energy.shape[1]:
        raise Exception("nS_final > exciton_energy.shape[1] or nS_initial > exciton_energy.shape[1]")

    # (b) make task plan
    workload_over_kmap = int(kmap.shape[0] * kmap.shape[0])
    plan_list, start_time, start_time_proc = before_parallel_job(rk=rank,size=size,workload_para=workload_over_kmap)
    plan_list = comm.scatter(plan_list,root=0)

    # print('process_%d. plan is ' % rank, plan_list, 'workload:', plan_list[-1]-plan_list[0])
    # sys.stdout.flush()

    # (c) This is exph_mat for each process: G(Q_kmap,q_kmap,n,m,v), initialize time
    # progress = ProgressBar(kmap.shape[0]*kmap.shape[0], fmt=ProgressBar.FULL)
    exph_mat_each_process = np.zeros((kmap.shape[0],kmap.shape[0],nS_initial,nS_final,n_phonon))
    time_prc0_start = time.time()

    if rank == 0:
        print("process_%d. takes memory of %.2f MB"%(rank,sys.getsizeof(exph_mat_each_process)/8/1024/1024))
        print('estimated whole memory is %.2f MB'%(size * sys.getsizeof(exph_mat_each_process)/8/1024/1024))
        sys.stdout.flush()

    Qq_list_dic = Q_q_mesh(kmap.shape[0])

    for Qq_iterate in range(plan_list[0],plan_list[-1]):
        # progress bar for parallel
        progress_bar_parallel(rank=rank,iterate_para=Qq_iterate,total=plan_list[-1]-plan_list[0],start_time=time_prc0_start)

        Qq_index_set = Qq_list_dic[Qq_iterate]
        Q_kmap = Qq_index_set[0]
        q_kmap = Qq_index_set[1]
        for j_initial_S in range(nS_initial):  # loop over initial exciton state m
            for j_final_S in range(nS_final):  # loop over initial exciton state m
                for j_phonon in range(n_phonon):  # loop over phonon mode v

            # gkQ
                    exph_mat_each_process[Q_kmap, q_kmap, j_initial_S, j_final_S, j_phonon] = np.abs(gqQ(n_ex_acv_index=j_initial_S,
                                                                                                  m_ex_acv_index=j_final_S,
                                                                                                  v_ph_gkk=j_phonon,
                                                                                                  Q_kmap=Q_kmap,
                                                                                                  q_kmap=q_kmap,
                                                                                                  acvmat=acvmat,
                                                                                                  gkkmat=gkkmat,
                                                                                                  kmap=kmap,
                                                                                                  kmap_dic=kmap_dic,
                                                                                                  bandmap_occ=bandmap_occ,
                                                                                                  muteProgress=True))**2

    # after_parallel ...
    # exph_rcev_to_0 = comm.gather(exph_mat_each_process, root=0)
    # value = after_parallel_sum_job(rk=rank, size=size, receive_res=exph_rcev_to_0, start_time=start_time,
    #                                start_time_proc=start_time_proc)
    value = np.zeros_like(exph_mat_each_process)
    comm.Reduce(exph_mat_each_process,value,op=MPI.SUM,root=0)

    if rank == 0:
        f = h5.File('gqQ.h5','w')
        f['data'] = value
        f.close()

def progress_bar_parallel(rank,iterate_para,total,start_time):
    if rank == 0:
        if total < 10:
            divisor = 1
        else:
            divisor = round(total/10)

        if iterate_para % divisor == 0:  # progress bar for real calculation
            print("[EXPH-Mat calculation] progress of process_%d: %.2f" % (rank, 100 * iterate_para / total) + "%"+", time lasted: %.1f sec"%(time.time()-start_time))
            sys.stdout.flush()

def Q_q_mesh(number_point):
    count = 0
    Q_q_dic = {} # {0:(Q,q)}
    for Q in range(number_point):
        for q in range(number_point):
            Q_q_dic[count] = (Q,q)
            count += 1
    return Q_q_dic


if __name__ == "__main__":
    # res_para = para_gqQ(path='../',mute=True)
    # print('res_para:',res_para)
    gqQ_h5_generator_Para(nS_initial=10, nS_final=10 ,path="../",mute=True)
    pass