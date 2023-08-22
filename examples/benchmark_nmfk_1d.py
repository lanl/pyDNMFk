#To run this code please run `mpirun -n 4 python dist_pynmfk_1d_wtsi.py` in command line.
import sys
import pyDNMFk.config as config

config.init(0)
from pyDNMFk.pyDNMFk import *
from pyDNMFk.utils import *
from pyDNMFk.dist_comm import *
from scipy.io import loadmat


def dist_nmfk_1d_nnsvd_init_wtsi(nGPUs=None,nccl_comm=None, topology=None ):
    args = parse()
    args.use_gpu = True
    args.use_gpu = False
     
    #args.gpu_partition = 'auto'
    args.gpu_partition = 'col'
    #args.gpu_partition = 'row'
     
    args.IMAX = 14400 #4096 #1024 #2048 #512 #256
    args.JMAX = 4800 #9600 #4096 #1024 #2048 #512 #256

    args.A_Is_Larage = True
    args.A_Is_Larage = False

    #args.W_update = True
    #args.W_update = False
    args.init = 'rand'
    #args.init = 'nnsvd'
    rank = comm.rank
    size = comm.size
    if args.use_gpu:
       if nGPUs is None: nGPUs = size
    else:
       nGPUs = size

    args.fpath = '../data/'

    #args.fname,args.ftype = 'wtsi_57600_38400_8_1e-08', 'npy'
    args.fname,args.ftype = 'wtsi_57600_38400_8_1.0', 'npy'
    #args.fname,args.ftype = 'wtsi_57600_38400_8_1e-06', 'mat'

    if args.use_gpu:
        try:
            partition_type = args.gpu_partition.lower()
            if partition_type in ['row', 'row1', '1row' ,'row_1d', '1d_row']:
                args.gpu_partition = 'row_1d'
            elif partition_type in ['col', 'col1', '1col' ,'col_1d', '1d_col', 'column', 'column_1d', '1d_column']:
                args.gpu_partition = 'col_1d'
            elif partition_type in ['auto', 'optim', 'optimal']:
                args.gpu_partition = 'auto'
            else:
                raise Exception(f"[!!] GPU grid partition {args.gpu_partition} is not supported. Try 'auto', 'column', 'row'")
        except:
              args.gpu_partition = 'auto'   
    if args.gpu_partition in ['row_1d']:
        p_r, p_c = 1, nGPUs # Row
        METHOD = 'Row_1d'
    elif args.gpu_partition in ['col_1d']:
         p_r, p_c = nGPUs, 1  # Col
         METHOD = 'Col_1d'
    else:                     #AUTO
        if rank == 0: print(f'[!!][AUTO GPU PARTITION] Trying to find optimal aprtition...')
        try:
            NN= args.fname.split('_')
            if int(NN[1]) < int(NN[2]):
                p_r, p_c = 1, nGPUs # Row
                METHOD = 'Row_1d'
            else:
                p_r, p_c = nGPUs, 1  # Col
                METHOD = 'Col_1d'
            if rank == 0: print(f"[!!][AUTO GPU PARTITION] {METHOD} GPU partition selected") 
        except:
            print(f'[!!] USING DEFAULT METHOD')
            p_r, p_c, METHOD = nGPUs, 1, 'Col_1d'  # Col
            #p_r, p_c, METHOD = 1, nGPUs, 'Row_1d'  # Row

    comms = MPI_comm(comm, p_r, p_c)
    comm1 = comms.comm
    args.size, args.rank, args.comm, args.p_r, args.p_c = size, rank, comms, p_r, p_c
    args.row_comm, args.col_comm, args.comm1 = comms.cart_1d_row(), comms.cart_1d_column(), comm1
    rank = comms.rank
    #if rank == 0: print(f"Running {METHOD}") 
    args.start_k = 4
    args.end_k = 8
    #args.k = 4
    args.step = 1
    args.sill_thr = 0.6
    args.itr = 10
    args.verbose = True
    args.norm = 'fro'
    args.method = 'mu'
    args.precision = np.float32
    args.checkpoint = False
    A_ij = data_read(args).read().astype(args.precision)
    args.results_path = '../results/'
    m,n            =  A_ij.shape
    J              =  min(args.JMAX, n)
    args.nGPUs     = nGPUs
    args.grid_m    = m
    args.grid_m    = n
    args.grid_m    = J

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

dist_nmfk_1d_nnsvd_init_wtsi(nGPUs=size)
