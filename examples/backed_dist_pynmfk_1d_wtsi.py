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
    #args.use_gpu = False
     
    #args.gpu_partition = 'auto'
    args.gpu_partition = 'col'
    #args.gpu_partition = 'row'
     
    args.IMAX = 40960 #1024 #2048 #512 #256
    args.JMAX = 40960 #1024 #2048 #512 #256
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
    #args.fname = 'swim'
    #args.fname = 'wtsi'
    
    #args.fname = 'wtsi_1024_524288_3_1.0' #Row
    #args.fname = 'wtsi_524288_1024_3_1.0' #Col

    #args.fname = 'wtsi_1024_32768_3_1.0' #Row
    #args.fname = 'wtsi_32768_1024_3_1.0' #Col

    #args.fname = 'wtsi_4096_65536_5_1.0' #Row
    #args.fname = 'wtsi_65536_4096_5_1.0' #Col

    #args.fname = 'wtsi_1024_65536_5_1.0' #Row
    #args.fname = 'wtsi_65536_1024_5_1.0' #Col

    #args.fname = 'wtsi_1024_131072_5_1.0' #Row 
    #args.fname = 'wtsi_131072_1024_5_1.0' #Col

    #args.fname = 'wtsi_512_4096_5_1.0' #Row
    #args.fname = 'wtsi_4096_512_5_1.0'  #Col

    #args.fname = 'wtsi_4096_65536_5_1.0' #Row
    #args.fname = 'wtsi_65536_4096_5_1.0' #Col
    
    #args.fname = 'wtsi_1024_65536_8_1.0' #Row
    #args.fname = 'wtsi_65536_1024_8_1.0' #Col

    #args.fname = 'X_combined_scaled'

    #args.ftype = 'npy' #'mat'
    #args.ftype = 'npy' #'npy'

    args.fname,args.ftype = 'US_11260_15056', 'npy'

    try:
        NN= args.fname.split('_')
        if int(NN[1]) < int(NN[2]):
            p_r, p_c = 1, nGPUs # Row
            METHOD = 'Row_1d'
        else:
            p_r, p_c = nGPUs, 1  # Col
            METHOD = 'Col_1d'
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
    args.start_k = 2
    args.end_k = 20
    #args.k = 4
    args.step = 2
    args.sill_thr = 0.6
    args.itr = 1000
    args.verbose = True
    args.norm = 'fro'
    args.method = 'mu'
    args.precision = np.float32
    args.checkpoint = False
    A_ij = data_read(args).read().astype(args.precision)
    #print(f"[+] Data set: [{A_ij.shape[0]}x{A_ij.shape[1]}] ({A_ij.dtype})")
    #print(f"[+] info = {np.finfo(A_ij.dtype).eps}")
    args.results_path = '../results/'
    m,n            =  A_ij.shape
    J              =  min(16*1024, n)
    args.nGPUs     = nGPUs
    args.grid_m    = m
    args.grid_m    = n
    args.grid_m    = J

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

dist_nmfk_1d_nnsvd_init_wtsi(nGPUs=size)
