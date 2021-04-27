#To run this code please run `mpirun -n 4 python dist_pynmfk_1d_wtsi.py` in command line.


import sys


import pyDNMFk.config as config

config.init(0)
from pyDNMFk.pyDNMFk import *
from pyDNMFk.utils import *
from pyDNMFk.dist_comm import *
from scipy.io import loadmat

def dist_nmfk_1d_nnsvd_init_wtsi():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    p_r, p_c = 4, 1
    comms = MPI_comm(comm, p_r, p_c)
    comm1 = comms.comm
    rank = comm.rank
    size = comm.size
    args = parse()
    args.size, args.rank, args.comm, args.p_r, args.p_c = size, rank, comms, p_r, p_c
    args.row_comm, args.col_comm, args.comm1 = comms.cart_1d_row(), comms.cart_1d_column(), comm1
    rank = comms.rank
    args.fpath = '../data/'
    args.fname = 'wtsi'
    args.ftype = 'mat'
    args.start_k = 2
    args.end_k = 6
    args.step = 1
    args.sill_thr = 0.6
    args.itr = 1000
    args.init = 'nnsvd'
    args.verbose = True
    args.norm = 'fro'
    args.method = 'mu'
    args.precision = np.float32
    A_ij = data_read(args).read().astype(args.precision)
    args.results_path = '../Results/'
    nopt = PyNMFk(A_ij, factors=None, params=args).fit()
    assert nopt == 4


dist_nmfk_1d_nnsvd_init_wtsi()

