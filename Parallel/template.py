from mpi4py import MPI

# 1.0 initialization:
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()