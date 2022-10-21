from ELPH.EX_PH_mat import gqQ
from Parallel.Para_common import plan_maker
from mpi4py import MPI
from IO.IO_common import construct_kmap, read_kmap
from time import  process_time
import time

# todo:!!!

def para_over_summation(workload, job_function, **kwargs):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()