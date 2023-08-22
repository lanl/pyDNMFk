# @author: Ismael Boureima
import sys
from mpi4py import MPI

# Attempt to import the cupy library for GPU array operations.
try:
    import cupy as cp
except ImportError:
    print("Unable to import Cupy.", file=sys.stderr)
    cp = None

# Attempt to import NCCL for GPU-to-GPU communication support from cupy.
try:
    from cupy.cuda import nccl
except ImportError:
    import sys
    print("NCCL is not supported.", file=sys.stderr)
    nccl = None


def get_NCCL_unique_id():
    """
    Fetch a unique ID for NCCL group communication.

    Returns:
        - Unique ID if NCCL is available, otherwise None.
    """
    if nccl is not None: return nccl.get_unique_id()



class MPIComm():
    """
    Class to handle basic MPI communication.

    This class acts as a simple wrapper around basic MPI communication
    functions for simplification and clarity.
    """
    def __init__(self):
        self.comm  = MPI.COMM_WORLD
        self.size  = self.comm.Get_size()
        self.rank  = self.comm.Get_rank()

    def Allreduce(self, send_arr, op=MPI.SUM, stream=None):
        """
        Perform an all-reduce operation across all MPI processes.

        Args:
            send_arr (array-like): Data to be reduced.
            op (MPI.Op): The reduction operation to be applied. Default is MPI.SUM.
            stream: CUDA stream for async operations (currently unused).

        Returns:
            array-like: The reduced array.
        """

        return self.comm.allreduce(send_arr)

    def Reduce(self, send_arr, op=MPI.SUM, root=0, stream=None):
        """
        Perform a reduce operation over MPI processes.

        Args:
            send_arr (array-like): Data to be reduced.
            op (MPI.Op): The reduction operation to be applied. Default is MPI.SUM.
            root (int): Rank of the root process. Default is 0.
            stream: CUDA stream for async operations (currently unused).
        """
        self.comm.reduce(send_arr, op, root=root)

    def Bcast(self, arr, root=0, stream=None):
        """
        Broadcast data to all MPI processes.

        Args:
            arr (array-like): Data to be broadcasted.
            root (int): Rank of the root process initiating the broadcast. Default is 0.
            stream: CUDA stream for async operations (currently unused).
        """
        self.comm.bcast(arr, root=root)




class NCCLComm(nccl.NcclCommunicator):
    """
    Class to handle NCCL GPU communication.

    This class acts as a wrapper around NCCL's communication functions
    optimized for GPU-to-GPU communications.
    """
    def __init__(self, ndev, commId, rank):
        """
        Args:
            ndev (int): Number of devices (GPUs) involved in the communication.
            commId (ncclUniqueId): A unique NCCL ID for group communication.
            rank (int): Rank of the current process within the group.
        """
        super(NCCLComm, self).__init__(ndev, commId, rank)
        self.comm   = nccl.NcclCommunicator
        self.size   = ndev
        self.rank   = rank
        self.commId = commId


    def get_unique_id(self):
        """
        Returns:
            ncclUniqueId: The unique NCCL ID for this communicator.
        """
        return self.commId

    def get_NCCL_count_dtype(self, arr):
        """
        Determine the data count and data type for NCCL communication based on array dtype.
        """
        if arr.dtype == cp.complex64:
            return arr.size*2, nccl.NCCL_FLOAT32
        elif arr.dtype == cp.complex128:
            return arr.size*2, nccl.NCCL_FLOAT64
        elif arr.dtype == cp.float32:
            return arr.size, nccl.NCCL_FLOAT32
        elif arr.dtype == cp.float64:
            return arr.size, nccl.NCCL_FLOAT64
        else:
            raise ValueError("This dtype is not supported by NCCL.")

    def Allreduce(self, send_arr, recv_arr, op=nccl.NCCL_SUM, stream=None):
        """
        Perform an all-reduce operation across all GPU processes using NCCL.
        """
        sendbuf = send_arr.__cuda_array_interface__['data'][0]
        recvbuf = recv_arr.__cuda_array_interface__['data'][0]
        count, datatype = self.get_NCCL_count_dtype(send_arr)
        if stream is None:
            stream = cp.cuda.Stream.null.ptr
        else:
            stream = stream.ptr
        super(NCCLComm, self).allReduce(sendbuf, recvbuf, count, datatype, op, stream)

    def Reduce(self, send_arr, recv_arr, op=nccl.NCCL_SUM, root=0, stream=None):
        """
        Perform a reduce operation across all GPU processes using NCCL.
        """
        sendbuf = send_arr.__cuda_array_interface__['data'][0]
        recvbuf = recv_arr.__cuda_array_interface__['data'][0]
        count, datatype = self.get_NCCL_count_dtype(send_arr)
        if stream is None:
            stream = cp.cuda.Stream.null.ptr
        else:
            stream = stream.ptr
        super(NCCLComm, self).reduce(sendbuf, recvbuf, count, datatype, op, root, stream)

    def Bcast(self, arr, root=0, stream=None):
        """
        Perform broadcast operation from root to all GPU processes using NCCL.
        """
        buff = arr.__cuda_array_interface__['data'][0]
        count, datatype = self.get_NCCL_count_dtype(arr)
        if stream is None:
            stream = cp.cuda.Stream.null.ptr
        else:
            stream = stream.ptr
        super(NCCLComm, self).bcast(buff, count, datatype, root, stream)





class SUPER_COMM():
    """
    A unified communicator that supports both MPI (for CPUs) and NCCL (for GPUs).
    """
    def __init__(self, ndev=None, commId=None, rank=None, backend="MPI"):
        """
        Args:
            ndev (int, optional): Number of devices (GPUs) involved in the communication.
            commId (ncclUniqueId, optional): A unique NCCL ID for group communication.
            rank (int, optional): Rank of the current process within the group.
            backend (str): Communication backend to use. Can be "MPI" or "NCCL".
        """
        self.backend = backend.upper()
        if self.backend in ["MPI","CPU","DEFAULT"]:
            self.comm = MPI_COMM()
        elif self.backend in ["NCCL","GPU","CUDA","BEST","PERFORMANCE"]:
            assert ndev is not None, "[!][COMM ERROR] SUPER_COMM NCCL backend requires valid ndev"
            assert commId is not None, "[!][COMM ERROR] SUPER_COMM NCCL backend requires valid commID"
            assert rank is not None, "[!][COMM ERROR] SUPER_COMM NCCL backend requires valid rank"
            self.comm = NCCL_COMM(ndev=ndev, commId=commId, rank=rank)
            #self.comm = NCCLComm(ndev=ndev, commId=commId, rank=rank)
        else:
            raise "[!][COMM ERROR] Backend {} not supported".format(self.backend)
        self.size    = self.comm.size
        self.rank    = self.comm.rank
