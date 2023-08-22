from mpi4py import MPI
comm        = MPI.COMM_WORLD
size        = comm.Get_size()
rank        = comm.Get_rank()

import cupy as cp
from cupy.cuda import nccl
from pyDNMFk.communicators import NCCLComm
from pyDNMFk.comm_utils import GetTopology
from pyDNMFk.toolz import log

cp.cuda.Device(rank).use()

topology = GetTopology(comm)

if rank == 0:
    global_uID = nccl.get_unique_id()
else:
    global_uID = None
global_uID = comm.bcast(global_uID, root=0)
comm.barrier()

log(f"[+] gID = {global_uID}", rank=rank, lrank=rank)

nccl_comm = NCCLComm(ndev=size, commId=global_uID, rank=rank)
