from mpi4py import MPI
import time
from time import process_time
from mpi4py import MPI
from Common.progress import ProgressBar
from Parallel.Para_common import plan_maker



#===================initialization=====================
# 1.0 initialization:
# (1) create comm
# (2) set plan_list as None for every one
# (3) load data for every proc todo: improve this
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
plan_list = None
a = list(range(12))
# a = list(range(30000000))
# a = list(range(30000))
# status = 'pending'
# print('process= %d. plan is '%rank, plan_list)
#-----------------------------------------------------
# print('process= %d. hello! '%rank)

#========================plan=========================
# 2.0 Plan before calculation
# (a) proc_0 makes plan for all procs (including proc_0)

if rank == 0:
    # status = 'start'
    start_time = time.time()
    start_time_proc = process_time()

    #method 1>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # proc_0 make plan for all calculations
    # determine the loop_start and loop_end
    # workload_per_proc = len(a) // size
    # workload_left = len(a)%size
    # plan_list = []
    # if workload_left != 0:
    #     print('warning: proc is noe divided evenly')
    # for i in range(size):
    #     if i != size - 1:
    #         plan_list.append([i*workload_per_proc, (i+1)*workload_per_proc])
    #     else:
    #         plan_list.append([i*workload_per_proc,len(a)])
    # print('proc_0 is speaking: work plan is like:', plan_list)
    #method 1>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    plan_list = plan_maker(nproc=size, nloop=len(a))
    print(plan_list)
else:
    pass
# (b) distribute plan
plan_list = comm.scatter(plan_list,root=0)
# print('process= %d. plan is '%rank, plan_list)
#-----------------------------------------------------


#======================calculation=====================
# 3.0 calculation
# (a) each proc is doing job!
# (b) progress
res = 0
for i in range(plan_list[0],plan_list[-1]):
    for j in range(10000000):
        # if i%100000==0:
        #     print(i,'/',-1*plan_list[0]+plan_list[-1])
        res = res + a[i]
# print('process= %d. res is '%rank, res)
print('process= %d. res is '%rank, res)

# (c) collect result from each processor
res_rcev = comm.gather(res,root=0)


#======================collect========================
if rank == 0:
    value = 0
    print('===================================')
    print('process= %d is summarizing ' % rank)
    for k in range(size):
        value = value + res_rcev[k]
    print("sub_res is", res_rcev)
    print("res is",value)
    end_time = time.time()
    end_time_proc = process_time()
    print("the wall time is: %.3f s"%(end_time - start_time))
    print("the proc time is: %.3f s"%(end_time_proc - start_time_proc))
    print('===================================')
#-----------------------------------------------------




# end_time = time.time()

# print("the running time is: %.3f s"%(end_time - start_time))
# print(res)


