from __future__ import division, print_function
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
rank = comm.Get_rank()


scal = None
mat = np.empty([3, 3], dtype='d')

arr = np.empty(5, dtype='d')
result = np.empty(5, dtype='d')


if rank == 0:
    scal = 55.0
    mat[:] = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    arr = np.ones(5)
    result = 2*arr

comm.Bcast([result, MPI.DOUBLE], root=0)
scal = comm.bcast(scal, root=0)
comm.Bcast([mat, MPI.DOUBLE], root=0)

print("Rank: ", rank, ". Array is:\n", result)
print("Rank: ", rank, ". Scalar is:\n", scal)
print("Rank: ", rank, ". Matrix is:\n", mat)
