from mpi4py import MPI
import time


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# initial
# status = 'pending'
plan_list = None
print('process= %d. plan is '%rank, plan_list)


#=============loaded data for every proc==============
# a = list(range(30000000))
# a = list(range(30000))
a = list(range(60000000))
#=====================================================

if rank == 0:
    # status = 'start'
    start_time = time.time()

    # proc_0 make plan for all calculations
    # determine the loop_start and loop_end
    workload_per_proc = len(a) // size
    workload_left = len(a)%size
    plan_list = []
    if workload_left != 0:
        print('warning: proc is noe divided evenly')
    for i in range(size):
        if i != size - 1:
            plan_list.append([i*workload_per_proc, (i+1)*workload_per_proc])
        else:
            plan_list.append([i*workload_per_proc,len(a)])
    print('proc_0 is speaking: work plan is like:', plan_list)

else:
    pass


plan_list = comm.scatter(plan_list,root=0)
print('process= %d. plan is '%rank, plan_list)

# if rank == 5:
#     plan_list = None
#
# if plan_list != None:
#     status = 'normal'
#     pass
# else:
#     raise Exception("process %d does not get task!"%rank)
#     status
# start your work!

res = 0
for i in range(plan_list[0],plan_list[-1]):
    res = res + a[i]

print('process= %d. res is '%rank, res)


res = comm.gather(res,root=0)
if rank == 0:
    value = 0
    print('process= %d is summarizing ' % rank)
    for i in range(size):
        value = value + res[i]
    print("sub_res is", res)
    print("res is",value)
    end_time = time.time()
    print("the running time is: %.3f s"%(end_time - start_time))






# end_time = time.time()

# print("the running time is: %.3f s"%(end_time - start_time))
# print(res)


