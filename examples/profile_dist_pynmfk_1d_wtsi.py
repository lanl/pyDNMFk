#To run this code please run `mpirun -n 4 python dist_pynmfk_1d_wtsi.py` in command line.
import sys
import pyDNMFk.config as config

config.init(0)
from pyDNMFk.pyDNMFk import *
from pyDNMFk.utils import *
from pyDNMFk.dist_comm import *
from scipy.io import loadmat


def dist_nmfk_1d_nnsvd_init_wtsi(nGPUs=None,nccl_comm=None, topology=None ):
    if nGPUs is None: nGPUs = size

    p_r, p_c = nGPUs, 1
    comms = MPI_comm(comm, p_r, p_c)
    comm1 = comms.comm
    rank = comm.rank
    size = comm.size
    args = parse()
    args.size, args.rank, args.comm, args.p_r, args.p_c = size, rank, comms, p_r, p_c
    args.row_comm, args.col_comm, args.comm1 = comms.cart_1d_row(), comms.cart_1d_column(), comm1
    args.nccl_comm = nccl_comm
    args.topology = topology
    rank = comms.rank
    args.fpath = '../data/'
    args.fname = 'wtsi'
    args.ftype = 'mat'
    args.start_k = 4
    args.end_k = 8
    args.step = 4
    args.sill_thr = 0.6
    args.itr = 4 #1000
    args.verbose = True
    args.norm = 'fro'
    args.method = 'mu'
    args.precision = np.float32
    args.checkpoint = False
    A_ij = data_read(args).read().astype(args.precision)
    args.results_path = '../results/'
    args.use_gpu = True
    #args.k = 4
    #args.W_update = True
    #args.W_update = False
    args.init = 'rand'
    #args.init = 'nnsvd'
    #nopt = PyNMF(A_ij, factors=None, params=args).fit()
    nopt = PyNMFk(A_ij, factors=None, params=args).fit()
    #assert nopt == 4

from mpi4py import MPI
comm        = MPI.COMM_WORLD
size        = comm.Get_size()
rank        = comm.Get_rank()

import cupy as cp
from cupy.cuda import nccl
from pyDNMFk.communicators import NCCLComm
from pyDNMFk.comm_utils import GetTopology

topology = GetTopology(comm)
cp.cuda.Device(topology.lrank).use()

if rank == 0:
    global_uID = nccl.get_unique_id()
else:
    global_uID = None
global_uID = comm.bcast(global_uID, root=0)
comm.barrier()
nccl_comm = NCCLComm(ndev=comm.size, commId=global_uID, rank=comm.rank)

dist_nmfk_1d_nnsvd_init_wtsi(nGPUs=size, nccl_comm=nccl_comm, topology=topology)
