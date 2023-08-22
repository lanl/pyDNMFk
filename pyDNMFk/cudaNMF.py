# @author: Ismael Boureima
# Necessary imports for the module
import os, time
import numpy as np
import numpy
from queue import Queue
from .toolz import log, blue, green, red, amber

# Try importing cupy modules for GPU support
try:
    import cupy as cp
    import cupy, cupyx
except:
    raise Exception(f"[{red('!!')}] Unable to import Cupy")

# Try importing necessary cupy backend modules
try:
    from cupy.cuda import nccl
except:
    raise Exception(f"[{red('!!')}] Unable to import NCCL Cupy backend")

# Importing specific functions from cupy to work with
from cupy import asarray  as _asarray
from cupy import asnumpy  as _asnumpy
from cupy import divide   as _divide
from cupy import matmul   as _matmul
from cupy import multiply as _multiply
from cupy import zeros    as _zeros

# Additional imports for the module
from .communicators import NCCLComm
from .comm_utils import GetTopology
from mpi4py import MPI
from .cupyCuSPARSELib  import spMM as _spMM
from .cupyCuSPARSELib  import spMat, spRandMat, spMM
from .cupy_utils       import pin_memory as _pin_mem
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, dia_matrix, issparse
import scipy
from .utils import data_operations


# Defining some constants for the module
PRECISIONS  = {'float32':np.float32, 'float64':np.float64, 'int16':np.int16, 'int32':np.int32,'int64':np.int64}
WORKLOADS   = {'nmf':'NMF', 'nmfk':'NMFk', 'bench':'BENCHMARK' }
NMF_NORMS   = {'fro':'FROBENUS', 'kl':'KL'}
NMF_METHODS = {'mu':'Multiplicative Update'}
NMF_INITS   = {'rand':'RANDOM', 'random':'RANDOM', 'nnsvd':'Non-Negative SVD', 'svd':'SVD'}

# Utility function to raise an exception for unsupported operations
def noSupportFor(X):
    raise Exception(f"[{red('!!')}] {amber(X)} is not yet supported on GPU")


# The main class to perform NMF on GPU
class cudaNMF():

    def __init__(self, A_ij,k, params, factors=None, save_factors=False):
        r"""
         Initialize the cudaNMF class.

         Parameters
         ----------
         A_ij : array-like
             The matrix to be factorized.
         k : int
             Rank of the factorization.
         params : dict
             Configuration parameters for NMF.
         factors : array-like, optional
             Initial guess for the factors.
         save_factors : bool, optional
             Whether to save factors or not.

         """
        self.A_ij          = A_ij
        self.params        = params
        self.k             = k
        
        self.factors       = factors
        self.save_factors  = save_factors
        self.VRBZ          = params.verbose
        try: 
            self.comm       = params.comm1
        except:
            print(f"{149*'#'}")
            print("[{}]                                          --[{}  comm {}]--".format(red("!"),blue('MPI'), red("NOT FOUND")))
            self.comm       = MPI.COMM_WORLD
            self.rank       = self.comm.rank
            if self.rank == 0:
                print(f"{149*'#'}")
                log("[{}]                                               --[{}  comm built {}]--".format(green("+"),blue('MPI'), green("OK")))
        self.rank       = self.comm.rank
        self.eps        = np.finfo(self.A_ij.dtype).eps
        self.params.eps = self.eps
        self.precision = self.params.precision
        self.topo_dim  = self.params.topo_dim
        self.local_m,self.local_n   = self.A_ij.shape
        self.p_r, self.p_c = self.params.p_r, self.params.p_c
        #[1] Identify distributed system's topology
        self.NCCL_COMM_BUILT        =  False
        self.identify_distributed_system_compute_topology()
        #[2] Identify Global compute grid parameters
        self.data_op = data_operations(self.A_ij, self.params)
        self.params = self.data_op.params
        self.GPU_GRID_BUILT         = False
        #[3] Identify A matrix partition method
        self.find_optimat_partition_strategy()
        #[4] Build GPU communication COMM
        self.checkGPUContext()
        self.buildSubCommunicators()
        METHOD = f"NMF_{self.GPU_PARTITION}_cuda"
        #[5] Check format (dense/sparse) of A matrix
        self.SPARSE_BUFF_PINNED_MEM_PARAMS_CONFIGURED = False
        self.check_A_matrix_format()
        if self.A_IS_SPARSE:
            METHOD = f"sparse_{METHOD}"
        else:
            METHOD = f"dense_{METHOD}"
        self.NMFMethod = METHOD
        #[6] Identifiy arrays that will be cached on GPU
        self.A_IS_BATCHED = self.params.A_is_batched
        self.A_is_cached,self.W_is_cached,self.H_is_cached  = None, None, None
        try:
            gpu_arrays = params.GPU_arrays
        except:
            if self.rank == 0: 
                self.log(msg = f"[!][CACHING] GPU arrays unspecified")
                self.log(msg = f"[!][CACHING] Assuming A, W, and H are cached on GPU")
            gpu_arrays = []
        self.gpu_arrays = []
        self.automatic_gpu_array_selection = False
        for ARR in gpu_arrays:
            arr = ARR.lower()
            if arr in ['auto', 'optim']:
                self.automatic_gpu_array_selection = True
            else:
                self.gpu_arrays.append(arr)
        if self.automatic_gpu_array_selection:
            if self.rank == 0: self.log(msg = f"[{red('!')}][{amber('COMING SOON')}] Automatic/Optimal GPU array selection is {red('Not yet supported')}")
        #[7] Build GPU MEM grid
        self.nBatch               = self.params.nBatch
        self.batchQeueSize        = self.params.batchQeueSize
        self.qs                   = self.batchQeueSize
        self.SQ                   = None
        self.MAXSTREAMS           = self.params.MAXSTREAMS
        if not self.GPU_GRID_BUILT: self.get_gpu_grid_params()
        self.BATCHING_AXIS        = None
        self.COL_BATCHING_AXIS    = None
        self.ROW_BATCHING_AXIS    = None
        self.find_optimal_baching_axis()
        self.set_batching_params()
        if self.A_IS_BATCHED: 
            self.NMFMethod = "{}_BATCHED".format(self.NMFMethod)
        else:
            self.cache_A_on_device()
        self.X_per_initialized = False
        self.show_A_info()
        #[8] Prepare events
        self.GB = 1024**3
        self.TB = 1024*self.GB
        self.dt = {'NMF':0.0, 'W_up':0.0, 'H_up':0.0, 
                   "allRed_XHT":0.0, "allRed_WHHT":0.0,
                   'allRed_WTX':0.0, 'allRed_WTWH':0.0,
                   'allRed_WTW':0.0, 'allRed_HHT':0.0,
                   'H2D_A':0.0, 'H2D_H':0.0, 'H2D_W':0.0,
                   'D2H_A':0.0, 'D2H_H':0.0, 'D2H_W':0.0,
                   'units':'[ms]'}
        self.events                             = {}
        self.events['start']                    = cp.cuda.Event()
        self.events['end']                      = cp.cuda.Event()
        self.events['h2d_s']                    = cp.cuda.Event()
        self.events['h2d_e']                    = cp.cuda.Event()
        self.events['d2h_s']                    = cp.cuda.Event()
        self.events['d2h_e']                    = cp.cuda.Event()
        self.events['reduce_s']                 = cp.cuda.Event()
        self.events['reduce_e']                 = cp.cuda.Event()
        self.events['nmf_start']                = cp.cuda.Event()
        self.events['nmf_end']                  = cp.cuda.Event()
        #[9] Build random generation initial state seeds # This is used so that cofactors are initialized
        #    locally and therefor avoiding broadcat communications
        if self.rank==0:
            iter_seeds = np.random.randint(100*self.params.end_k, size=2*self.params.end_k)
        else:
            iter_seeds = None
        iter_seeds=self.comm.bcast(iter_seeds, root=0)
        self.iter_seeds = iter_seeds
        #[10] Configure Pinned Memory management Pool
        #self.PINNEDMEMPOOL  = cp.cuda.PinnedMemoryPool()
        #cp.cuda.set_pinned_memory_allocator(self.PINNEDMEMPOOL.malloc)
        if self.A_IS_BATCHED:
            self.MEMPOOL        = cp.get_default_memory_pool()
            self.PINNEDMEMPOOL  = cp.cuda.PinnedMemoryPool()
            cp.cuda.set_pinned_memory_allocator(self.PINNEDMEMPOOL.malloc)
        else:
            self.MEMPOOL        = cp.get_default_memory_pool()
            self.PINNEDMEMPOOL  = cp.get_default_pinned_memory_pool()
        self.showMemStats()
        #[11] Build Qeues of CUDA STREAMS for Async copies management
        self.STREAMS        = {}
        self.stream         = []
        self.FREE_STREAM    = {}
        self.REDUCE_STREAM  = cp.cuda.stream.Stream()
        self.MAXSTREAMS     = min(self.nBatch, self.MAXSTREAMS)
        for n in range(self.MAXSTREAMS):
            self.stream.append("s%04d" %n)
            self.STREAMS[self.stream[-1]]    = cp.cuda.stream.Stream()
            self.FREE_STREAM[self.stream[-1]] = cp.cuda.Event()


    def log(self,msg):
        r"""
        Prints a log message.

        Parameters
        ----------
        msg : str
            The message to be logged.

        """
        log(msg = msg, rank=self.rank, lrank=self.lrank)


    def isCupyObject(self,x):
        r"""
        Checks if the input object is a Cupy object.

        Parameters
        ----------
        x : object
            The object to be checked.

        Returns
        -------
        bool
            True if x is a Cupy object, False otherwise.

        """
        return type(x) in [cp._core.core.ndarray]


    def getArrayType(self,x):
        r"""
        Returns the type of the array (CPU/GPU, dense/sparse, etc.).

        Parameters
        ----------
        x : array-like
            The array whose type needs to be determined.

        Returns
        -------
        str
            Type of the array (e.g., "CPU_DENSE", "GPU_SPARSE").

        """
        T = type(x)
        if T in [list]:
            return "LIST[{}]".format(self.getObjectType(x[0]))
        elif T in [np.ndarray]:
            return "CPU_DENSE"
        elif T in [cp._core.core.ndarray]:
            return "GPU_DENSE"
        elif T in [scipy.sparse.coo.coo_matrix, scipy.sparse.csr.csr_matrix, scipy.sparse.csc.csc_matrix]:
            return "CPU_SPARSE"
        elif T in [cupyx.scipy.sparse.coo.coo_matrix, cupyx.scipy.sparse.csr.csr_matrix, cupyx.scipy.sparse.csc.csc_matrix]:
            return "GPU_SPARSE"
        else:
            return "UNKNOWN"


    def identify_distributed_system_compute_topology(self):
        r"""
        Identifies the distributed system's compute topology and sets up the necessary
        attributes related to topology.
        """
        self.topology = GetTopology(self.comm)
        if self.VRBZ : self.show_topology()
        if self.rank == 0:
            global_uID = nccl.get_unique_id()
        else:
            global_uID = None
        self.global_uID = self.comm.bcast(global_uID, root=0)
        self.comm.barrier()
        self.myHost                 = self.topology.myHost
        self.lrank                  = self.topology.lrank
        self.nGPU_local             = self.topology.nGPU_local
        self.nGPUs                  = self.topology.topology['nGPUs']


    def compute_global_dim(self): ##[IDB]: This is needed here
        r"""Computes global dimensions m and n from given chunk sizes for any grid configuration"""
        if self.p_r != 1 and self.p_c == 1:
            self.global_n = self.local_n
            self.global_m = self.comm.allreduce(self.local_m)
        elif self.p_c != 1 and self.p_r == 1:
            self.global_n = self.comm.allreduce(self.local_n)
            self.global_m = self.local_m
        else:
            if self.rank % self.p_c == 0:
                self.global_m = self.local_m
            else:
                self.global_m = 0
            self.global_m = self.comm.allreduce(self.global_m)
            if self.rank // self.p_c == 0:
                self.global_n = self.local_n
            else:
                self.global_n = 0
            self.global_n = self.comm.allreduce(self.global_n)
            self.comm.barrier()


    def checkGPUContext(self):
        r"""
        Determines if the current instance is operating within a GPU context and sets up
        GPU-related attributes.
        """
        if self.lrank < self.nGPU_local:
            self.IN_GPU_CONTEXT = True
            self.lID            = self.lrank              # Local  GPU ID
            self.gID            = self.topology.gID       # Global GPU ID
            cp.cuda.Device(self.lrank).use()
            #self.device         = cp.cuda.Device()
            self.cudaDevice     = cp.cuda.runtime.getDevice()
            if self.VRBZ: self.log(msg=f"I am using <Global Device{self.gID}> | <Local Device{self.cudaDevice}> on {self.myHost}")
        else:
            self.IN_GPU_CONTEXT = False
            self.lID            = None
            self.gID            = None
            if self.VRBZ: self.log(msg = f"I am not using CUDA")
        self.comm.barrier()
        self.nccl_group_size = self.comm.allreduce(int(self.IN_GPU_CONTEXT), op=MPI.SUM)
        if self.rank == 0: self.log(msg = f"[+][NCCL] GROUP size = {self.nccl_group_size}")
        

    def check_A_matrix_format(self):
        r"""
        Checks the format of matrix A, decides if it's sparse or dense, and sets the
        necessary attributes related to the matrix's format.
        """
        if type(self.A_ij) in [scipy.sparse.coo.coo_matrix, scipy.sparse.csr.csr_matrix, scipy.sparse.csc.csc_matrix]:
            if self.params.densify_A:           # Dense A and Array format
                self.A_ij = self.A_ij.toarray() # Converting to dense for now (Fix Not Efficient)
                self.A_in_sparse_format = False
                self.A_IS_SPARSE        = False
            else:                               # Sparse A and sparse format
                self.A_in_sparse_format = True
                self.A_IS_SPARSE        = True
                self.A_nnz              =  self.A_ij.nnz
                self.A_density          = self.A_nnz/(self.global_m*self.global_n)
                self.MM = _spMM
        else:
            self.A_in_sparse_format = False
        if not self.A_in_sparse_format:
            self.A_nnz = np.count_nonzero(self.A_ij)
            self.A_nnz = self.comm.allreduce(self.A_nnz)
            self.A_density = self.A_nnz/(self.global_m*self.global_n)
            if self.A_density > self.params.sparsity_tresh:      # Dense A and Array format
                #self.NMFMethod = 'dense_{}'.format(self.NMFMethod)
                self.A_IS_SPARSE = False
                self.MM = _matmul
            else:                                                   # Dense A and sparse format
                self.A_IS_SPARSE = True
                if self.params.sparsify_A:
                    self.A_ij = csr_matrix(self.A_ij)
                    self.A_in_sparse_format = True
                    self.MM = _spMM
                else:
                    self.A_IS_SPARSE = False
                    self.MM = _matmul
        self.A_IS_DENSE = not self.A_IS_SPARSE


    def show_A_info(self):
        r"""
        Logs and displays information about matrix A, including dimensions, density,
        and other related attributes.
        """
        if self.rank == 0:
            self.log(f"[{green('+')}] [   {blue('Local')}   ] A_ij[{amber(self.local_m)}, {amber(self.local_n)}] ({red(round(np.prod(self.A_ij.shape)*4/(1024**3), 4) )} {red('GB')}) built {green('OK')}")
            self.log(f"[{green('+')}] [{blue('Distributed')}] A_ij[{amber(self.global_m)}, {amber(self.global_n)}] ({red( round(self.global_m*self.global_n*4/(1024**3), 4 ) )} {red('GB')}) built {green('OK')}")
            self.log(f"[{green('+')}]  Density         = {self.A_density}")
            if self.COL_PARTITION:
                self.log(f"[{green('+')}]  Nr = nBatch     = {self.nBatch}   -> Batch Size = m/nBatch = {self.batch_size}")
            else:
                self.log(f"[{green('+')}]  Nc = nBatch     = {self.nBatch}   -> Batch Size = n/nBatch = {self.batch_size}")
            if self.last_chunk > 0 : self.log(f"[{amber('!')}]  Uneven split         -> Last Batch Size       = {self.last_chunk}")
            #self.log(f"[{green('+')}]    Nc        = nGPUs = {self.grid_Nc} -> J  = n/Nc = {self.grid_J}")
            self.log(f"[{green('+')}]  Batch Qeue size = {self.batchQeueSize}")
            print(f"{149*'#'}")
            self.log(f"[{green('+')}]  {blue('WORKLOAD')}                    :  {green(WORKLOADS[self.params.work_load.lower()])}")
            self.log(f"[{green('+')}]  {blue('NMF METHOD')}                  :  {green(NMF_METHODS[self.params.method.lower()])}")
            self.log(f"[{green('+')}]  {blue('NMF NORM')}                    :  {green(NMF_NORMS[self.params.norm.lower()])}")
            self.log(f"[{green('+')}]  {blue('NMF INIT')}                    :  {green(NMF_INITS[self.params.init.lower()])}")
            self.log(f"[{green('+')}]  {blue('Number of perturbation')}      :  {green(str(self.params.perturbations))}")
            self.log(f"[{green('+')}]  {blue('Iterations/Perturbation')}     :  {green( str(self.params.itr)) }")
            self.log(f"[{green('+')}]  {blue('Range k')}                     :  {green(str(self.params.start_k))} < k < {amber(str(self.params.end_k))}")
            self.log(f"[{green('+')}]  {blue('Delta k')}                     :  {green(str(self.params.step))}")
            self.log(f"[{green('+')}]  {blue('Sill Threshold')}              :  {green(str(self.params.sill_thr)) }")
            self.log(f"[{green('+')}]  {blue('Noise')}                       :  {green(str(self.params.noise_var)) }")
            self.log(f"[{green('+')}]  {blue('GPU Arrays')}                  :  {green(self.gpu_arrays)}")
            print(f"{149*'#'}")
            print(f"{149*'#'}")
        self.sub_comm.barrier()


    def show_topology(self):
        r"""
        Displays information about the system's topology.
        """
        self.topology.showTopology()
        self.comm.barrier()
        log(f"[+] topo_dim    = {self.params.topo_dim}", rank=self.rank)
        log(f"[+] myHost      = {self.topology.myHost}", rank=self.rank)
        log(f"[+] lrank       = {self.topology.lrank}", rank=self.rank)
        log(f"[+] nGPU_local  = {self.topology.nGPU_local}", rank=self.rank)
        log(f"[+] nGPUs       = {self.topology.topology['nGPUs']}", rank=self.rank)
        self.comm.barrier()


    def buildSubCommunicators(self, rebuildNCCLCOM=False):
        r"""
        Args:
            rebuildNCCLCOM (bool): Flag indicating if NCCL Communicator should be rebuilt.

        Builds sub-communicators for the distributed system. Optionally rebuilds the NCCL communicator.
        """
        ...
        if self.IN_GPU_CONTEXT:
          color = 55
        else:
          color = 77
        self.color = color
        self.comm.barrier()
        self.sub_comm = self.comm.Split(color, self.rank)
        self.comm.barrier()
        if ((not self.NCCL_COMM_BUILT) or (rebuildNCCLCOM)): self.buildNCCLSubCommunicator()
    

    def buildNCCLSubCommunicator(self):
        r"""
        Builds the NCCL sub-communicator for GPU context. Sets the related attributes.
        """
        if self.IN_GPU_CONTEXT:
          if self.rank == 0:
            if self.VRBZ: self.log(f"[+] Broadcasting NCCL global_UID")
            global_uID = nccl.get_unique_id()
          else:
            global_uID = None
          self.global_uID = self.comm.bcast(global_uID, root=0)
          if self.VRBZ: self.log(f"[+] NCCL global_UID received OK")
          self.comm.barrier()
          self.nccl_comm = NCCLComm(ndev=self.nccl_group_size, commId=self.global_uID, rank=self.rank)
        else:
            self.nccl_comm = None
        self.comm.barrier()
        self.NCCL_COMM_BUILT = True
        if self.rank == 0:
            print(f"{149*'#'}")
            self.log(f"[{green('+')}]                                                --[{blue('NCCL')} comm built {green('OK')}]--")
            print(f"{149*'#'}")
        self.comm.barrier()
    

    def get_gpu_grid_params(self):
        r"""
        Fetches or calculates GPU grid parameters based on the current setup.
        Sets the related attributes.
        """
        if self.GPU_GRID_BUILT: return
        self.grid_loc_m, self.grid_loc_n      =  self.local_m, self.local_n
        self.grid_glob_m, self.grid_glob_n    =  self.global_m, self.global_n
        if self.params.init.lower() in ['nnsvd','svd','singular']:
            if self.COL_PARTITION: # n>m
                self.grid_L = int(self.grid_glob_m/self.nGPUs)
                self.grid_q = int(self.grid_L/self.batch_size)
            else:
                self.grid_L = int(self.grid_glob_n/self.nGPUs)
                self.grid_q = int(self.grid_L/self.batch_size)



    def cache_A_on_device(self):
        r"""
        Caches matrix A on the device for faster access during GPU operations.
        """
        if not self.GPU_GRID_BUILT: self.get_gpu_grid_params()
        if self.IN_GPU_CONTEXT:
            self.A_d = []
            for b in range(self.nBatch):
                if self.COL_PARTITION:
                    i0, i1 = b*self.batch_size, (b+1)*self.batch_size
                    if self.A_IS_SPARSE:
                        self.A_d.append(spMat( self.A_ij[i0:i1, :], Format='csr'))
                    else:
                        self.A_d.append(_asarray(self.A_ij[i0:i1, :], dtype=self.precision))
                else:
                    j0, j1 = b*self.batch_size, (b+1)*self.batch_size
                    if self.A_IS_SPARSE:
                        self.A_d.append(spMat( self.A_ij[:, j0:j1], Format='csr'))
                    else:
                        self.A_d.append(_asarray(self.A_ij[:, j0:j1], dtype=self.precision))
        else:
            self.A_d = None
        self.comm.barrier()
        self.A_is_cached = True


    def cache_H_on_device(self):   # W.I.P
        r"""
        This method sets the flag indicating that the matrix H is cached on the device.
        """
        self.H_is_cached = True


    def cache_W_on_device(self):   # W.I,P
        r"""
        This method sets the flag indicating that the matrix W is cached on the device.
        """
        self.W_is_cached = True



    def find_optimat_partition_strategy(self):
        r"""
        This method determines the best GPU partitioning strategy based on the provided parameters.
        It supports partitioning in 'row', 'column' or 'auto' modes.
        """
        # Attempt to set partition type from parameters; default to 'auto' on failure.
        try:
            partition_type              = self.params.gpu_partition.lower()
        except:
            partition_type              = 'auto'

        # Check the partition type and set the GPU_PARTITION attribute accordingly.
        # Multiple alias names are supported for each partitioning strategy.
        if partition_type in ['row', 'row1', '1row' ,'row_1d', '1d_row']:
            self.GPU_PARTITION = 'row_1d'
        # ... (similar checks for other partition types) ...
        elif partition_type in ['col', 'col1', '1col' ,'col_1d', '1d_col', 'column', 'column_1d', '1d_column']:
            self.GPU_PARTITION = 'col_1d'
        # If 'auto' mode, optimize for grid partition.
        elif partition_type in ['auto', 'optim', 'optimal']:
            # ... (logic to automatically choose the partition type) ...
            self.GPU_PARTITION = 'auto'
            if self.rank == 0:
                log(msg = f"[{amber('!')}][PARTION]: GPU_PARTITION is set to {amber(self.GPU_PARTITION)}", rank=self.rank, lrank=self.lrank)
                log(msg = f"[{amber('.')}][PARTION]: Optimizing for grid partion ...", rank=self.rank, lrank=self.lrank)
            if self.global_m >= self.global_n:
                self.GPU_PARTITION = 'row_1d'
            else:
                self.GPU_PARTITION = 'col_1d'
            if self.rank == 0: log(msg = f"[{green('+')}][PARTION]: GPU_PARTITION is set to {green(self.GPU_PARTITION)}", rank=self.rank, lrank=self.lrank)
        else:
            raise Exception(f"[!!] GPU grid partition {red(params.gpu_partition)} is not supported. Try {green('auto')}, {green('column')}, {green('row')}")
        # Set ROW and COL partition flags based on GPU_PARTITION.
        if self.GPU_PARTITION in ['row_1d']:
            self.COL_PARTITION = True
        else:
            self.COL_PARTITION = False
        self.ROW_PARTITION = not self.COL_PARTITION



    def find_optimal_baching_axis(self):
        try:
            self.BATCHING_AXIS = self.params.batching_axis.lower()
        except:
            self.BATCHING_AXIS = 'auto'
        if  self.BATCHING_AXIS in ['row', 'row1', '1row' ,'row_1d', '1d_row']:
            self.BATCHING_AXIS = 'row_1d'
        elif self.BATCHING_AXIS in ['col', 'col1', '1col' ,'col_1d', '1d_col', 'column', 'column_1d', '1d_column']:
            self.BATCHING_AXIS = 'col_1d'
        elif self.BATCHING_AXIS in ['auto', 'optim', 'optimal']:
            self.BATCHING_AXIS = 'auto'
            if self.rank == 0:
                log(msg = f"[{amber('!')}][BATCHING]: STRATEGY is set to {amber(self.GPU_PARTITION)}", rank=self.rank, lrank=self.lrank)
                log(msg = f"[{amber('.')}][BATCHING]: Optimizing for BATCHING STRATEGY...", rank=self.rank, lrank=self.lrank)
            if self.grid_loc_m >= self.grid_loc_n:
                self.BATCHING_AXIS = 'row_1d'
            else:
                self.BATCHING_AXIS = 'col_1d'
            if self.rank == 0: log(msg = f"[{green('+')}][BATCHING]: STRATEGY is set to {green(self.BATCHING_AXIS)}", rank=self.rank, lrank=self.lrank)
        else:
            raise Exception(f"[!!] BATCHING STRATEGY {red(params.BATCHING_AXIS)} is not supported. Try {green('auto')}, {green('column')}, {green('row')}")
        if self.BATCHING_AXIS in ['row_1d']:
            self.ROW_BATCHING_AXIS = True
        else:
            self.ROW_BATCHING_AXIS = False
        self.COL_BATCHING_AXIS = not self.ROW_BATCHING_AXIS


    def set_batching_params(self):
        if self.ROW_BATCHING_AXIS:              # Row-wise batchingn
            self.batch_size                     = self.local_m//self.nBatch
            self.last_chunk                     = self.local_m%self.nBatch
            self.grid_I                         = self.batch_size  # min(self.params.IMAX, self.grid_glob_m)
            self.params.grid_I                  = self.grid_I
            self.grid_Nr, self.grid_Nc          = self.nBatch, 1   # int(self.grid_loc_m/self.grid_I), int(1*self.nGPUs)
            self.grid_J                         = self.local_n     # int(self.grid_loc_n/self.grid_Nc)
        else:                                   # Column-wise batching
            self.batch_size                     = self.local_n//self.nBatch
            self.last_chunk                     = self.local_n%self.nBatch
            self.grid_J                         = self.batch_size  # min(self.params.JMAX, self.grid_loc_n)
            self.params.grid_J                  = self.grid_J
            self.grid_Nr, self.grid_Nc          = 1,self.nBatch    # 1, int(self.grid_loc_n/self.grid_J)
            self.grid_I                         = self.local_m     # int(self.grid_loc_m/self.grid_Nr)


    def get_managed_stream_queue(self):
        """
        Initializes and returns a dictionary for managing stream queues for GPU operations.
        """
        SQ={}
        SQ['Queue']  = Queue(maxsize = self.qs)
        SQ['READY']  = {}
        #[1] En-queue working streams and set their states to ready to work
        for j in range(self.qs):
            i = j+1
            SQ['READY'][i] = cp.cuda.stream.Stream()
            SQ['Queue'].put(i)
        SQ['REDUCE']       = cp.cuda.stream.Stream()
        SQ['COMPLETED'] = []
        return SQ


    def configure_sparse_buffers_pinned_mem_params(self):
        r"""
        Configures the parameters for pinned memory buffers when dealing with sparse matrices.
        """
        if self.SPARSE_BUFF_PINNED_MEM_PARAMS_CONFIGURED: return
        assert self.BATCHING_AXIS is not None, "[!] PINNED MEM BUFFERS are used only for BATCHING"
        assert self.A_IS_SPARSE, "[!] SPARSE PINNED MEM BUFFERS are used only for BATCHING A in SPARSE FORMAT"
        self.buff_idx, self.buff_ptr = [], []
        self.sparse_vect_size = {'dat':[],'idx':[],'ptr':[]}
        self.sparseData_max_vect_size, self.sparseIdx_max_vect_size, self.sparsePtr_max_vect_size = 0,0,0
        for b in range(self.nBatch):
            #if self.ROW_PARTITION:
            if self.ROW_BATCHING_AXIS:
                i0, i1 = b*self.batch_size, (b+1)*self.batch_size
                self.sparse_vect_size['dat'].append(len(self.A_ij[i0:i1, :].data))
                self.sparse_vect_size['idx'].append(len(self.A_ij[i0:i1, :].indices))
                self.sparse_vect_size['ptr'].append(len(self.A_ij[i0:i1, :].indptr))
                self.sparseData_max_vect_size =  max(self.sparseData_max_vect_size, len(self.A_ij[i0:i1, :].data))
                self.sparseIdx_max_vect_size  =  max(self.sparseIdx_max_vect_size, len(self.A_ij[i0:i1, :].indices))
                self.sparsePtr_max_vect_size  =  max(self.sparsePtr_max_vect_size, len(self.A_ij[i0:i1, :].indptr))
            else:  # Col Partition
                j0, j1 = b*self.batch_size, (b+1)*self.batch_size
                self.sparse_vect_size['dat'].append(len(self.A_ij[:, j0:j1].data))
                self.sparse_vect_size['idx'].append(len(self.A_ij[:, j0:j1].indices))
                self.sparse_vect_size['ptr'].append(len(self.A_ij[:, j0:j1].indptr))
                self.sparseData_max_vect_size = max(self.sparseData_max_vect_size, len(self.A_ij[:, j0:j1].data))
                self.sparseIdx_max_vect_size  = max(self.sparseIdx_max_vect_size, len(self.A_ij[:, j0:j1].indices))
                self.sparsePtr_max_vect_size  = max(self.sparsePtr_max_vect_size, len(self.A_ij[:, j0:j1].indptr))
            #print("[b%04d] lenData, lenIdx, lenPtr = {},{},{}".format(self.sparseData_max_vect_size, self.sparseIdx_max_vect_size, self.sparsePtr_max_vect_size) %b)
        self.SPARSE_BUFF_PINNED_MEM_PARAMS_CONFIGURED = True




    def allocate_sparse_batch_buffers(self):
        r"""
        Allocate memory for sparse batch buffers.

        This method configures memory parameters if not already set. Based on the specified
        batching axis, it initializes the appropriate buffers to store the data.
        """
        # Check if memory parameters for sparse buffers are configured
        if not self.SPARSE_BUFF_PINNED_MEM_PARAMS_CONFIGURED:
            self.configure_sparse_buffers_pinned_mem_params()

        # Allocate memory based on column batching axis
        if self.COL_BATCHING_AXIS:
            self.H_d, self.X_d = [], []
            for n in range(self.batchQeueSize):
                self.H_d.append(cp.empty((self.k, self.batch_size), dtype=self.A_ij.dtype))
                self.X_d.append(cp.empty(self.sparseData_max_vect_size, dtype=self.A_ij.dtype))
                self.buff_idx.append(cp.empty(self.sparseIdx_max_vect_size, dtype=cp.int32))
                self.buff_ptr.append(cp.empty(self.sparsePtr_max_vect_size, dtype=cp.int32))
        else: # ROW_BATCHING_AXIS
            self.W_d, self.X_d = [], []
            for n in range(self.batchQeueSize):
                self.W_d.append(cp.empty((self.batch_size, self.k), dtype=self.A_ij.dtype))
                self.X_d.append(cp.empty(self.sparseData_max_vect_size, dtype=self.A_ij.dtype))
                self.buff_idx.append(cp.empty(self.sparseIdx_max_vect_size, dtype=cp.int32))
                self.buff_ptr.append(cp.empty(self.sparsePtr_max_vect_size, dtype=cp.int32))


    def allocate_dense_batch_buffers(self):
        r"""
        Allocate memory for dense batch buffers.

        Based on the specified batching axis, this method initializes the appropriate
        buffers to store dense data.
        """

        # Allocate memory based on column batching axis
        if self.COL_BATCHING_AXIS:
            self.H_d, self.X_d = [], []
            for n in range(self.batchQeueSize):
                self.H_d.append(cp.empty((self.k, self.batch_size), dtype=self.A_ij.dtype))
                #self.X_d.append(cp.empty((self.grid_loc_m, self.batch_size), dtype=self.A_ij.dtype))
                self.X_d.append(cp.empty((self.grid_glob_m, self.batch_size), dtype=self.A_ij.dtype))
        else: # ROW_BATCHING_AXIS
            self.W_d, self.X_d = [], []
            for n in range(self.batchQeueSize):
                self.W_d.append(cp.empty((self.batch_size, self.k), dtype=self.A_ij.dtype))
                #self.X_d.append(cp.empty((self.batch_size, self.grid_loc_n), dtype=self.A_ij.dtype))
                self.X_d.append(cp.empty((self.batch_size, self.grid_glob_n), dtype=self.A_ij.dtype))


    def allocate_gpu_batch_buffers(self):
        r"""
        Allocate memory for batch buffers on GPU.

        Depending on whether the data is sparse or dense, it calls the appropriate
        method to initialize the buffers.
        """
        # Check if the data is batched
        if self.A_IS_BATCHED:
            if self.A_IS_SPARSE:
                self.allocate_sparse_batch_buffers()
            else:
                self.allocate_dense_batch_buffers()

    def showMemStats(self, msg=None):
        r"""
        Display memory statistics.

        It logs information about used and available memory, including details about
        the pinned memory pool.

        Parameters:
            msg (str): Optional message to include in the log.
        """
        # Only process with rank 0 displays the memory stats
        if self.rank == 0:
            if msg is not None: log(msg = f"[{amber('!')}][MEM INFO] {msg}", rank=self.rank, lrank=self.lrank)
            # Log info based on whether the data is batched or not
            if self.A_IS_BATCHED:
                if msg is not None: log(msg = f"[{amber('!')}][MEM INFO] {msg}", rank=self.rank, lrank=self.lrank)
                log(msg = f"[{amber('!')}][PINNEDMEM]: FREE BLOCKS = { amber(self.PINNEDMEMPOOL.n_free_blocks()) }", rank=self.rank, lrank=self.lrank)
            else:
                log(msg = f"[{amber('!')}][MEMPOOL]: USED = { amber(self.MEMPOOL.used_bytes()/self.GB)} GB || TOTAL = { red(self.MEMPOOL.total_bytes()/self.GB) } GB", rank=self.rank, lrank=self.lrank)
                log(msg = f"[{amber('!')}][PINNEDMEM]: FREE BLOCKS = { amber(self.PINNEDMEMPOOL.n_free_blocks()) }", rank=self.rank, lrank=self.lrank)

    def sampleA(self, noise_var, method, seed=None):
        r"""
        Samples matrix A based on a specified method and noise variance.

        Parameters:
            - noise_var (float): Variance of the noise.
            - method (str): Sampling method, either 'uniform' or 'poisson'.
            - seed (int, optional): Random seed for reproducibility.

        Raises:
            - Exception if an unsupported sampling method is provided.
        """
        self.noise_var = noise_var
        self.seed = seed
        if self.seed != None:
            cp.random.seed(self.seed)
        self.sampling_method = method
        if self.sampling_method == 'uniform':
            self.randM()
        elif self.sampling_method == 'poisson':
            self.poisson()


    def randM(self):
        r"""
        Perturbs the elements of X by multiplying them with a uniform random number in the range (1-epsilon, 1+epsilon).
        """

        # Check if X_per has been initialized, if not, initialize the arrays
        if not self.X_per_initialized: self.X_per, self.X_idx, self.X_ptr = [], [], []
        #self.showMemStats(msg = " RandM() 0")

        # Free up memory blocks in the memory pool
        self.MEMPOOL.free_all_blocks()
        #self.showMemStats(msg = " RandM() MEMPOOL Freed OK")

        # Get the max available memory in bytes
        MAXMEM = self.MEMPOOL.total_bytes()*1.0
        if self.A_IS_BATCHED: #CPU Array
            for b in range(self.nBatch):
                if self.A_IS_SPARSE:
                    if self.ROW_BATCHING_AXIS:
                        i0, i1 = b*self.batch_size, (b+1)*self.batch_size
                        X = self.A_ij[i0:i1, :]
                    else:  # Col Axis
                        j0, j1 = b*self.batch_size, (b+1)*self.batch_size
                        X = self.A_ij[:, j0:j1]
                    M = np.random.random_sample(X.data.shape).astype(self.A_ij.dtype)
                    M = 2.0*self.noise_var * M + self.noise_var + 1.0
                    X.data *= M
                    if not self.X_per_initialized: # Initialize sparse array pinned mem buffers
                        self.X_per.append(_pin_mem(np.append(X.data,np.zeros(self.sparseData_max_vect_size-len(X.data), dtype=X.data.dtype))))
                        self.X_idx.append(_pin_mem(np.append(X.indices,np.zeros(self.sparseIdx_max_vect_size-len(X.indices), dtype=X.indices.dtype))))
                        self.X_ptr.append(_pin_mem(np.append(X.indptr,np.zeros(self.sparsePtr_max_vect_size-len(X.indptr), dtype=X.indptr.dtype))))
                    else:
                        #print(f"Before : type(X_per) = {self.X_per[b].dtype}")
                        self.X_per[b] = np.append(X.data,np.zeros(self.sparseData_max_vect_size-len(X.data), dtype=X.data.dtype)).astype(self.A_ij.dtype)
                        self.X_idx[b] = np.append(X.indices,np.zeros(self.sparseIdx_max_vect_size-len(X.indices), dtype=X.indices.dtype))
                        self.X_ptr[b] = np.append(X.indptr,np.zeros(self.sparsePtr_max_vect_size-len(X.indptr), dtype=X.indptr.dtype))
                        #print(f"Afterpe : type(X_per) = {self.X_per[b].dtype}")

                else: # A in Dense
                    if self.ROW_BATCHING_AXIS:
                        i0, i1 = b*self.batch_size, (b+1)*self.batch_size
                        M = np.random.random_sample(self.A_ij[i0:i1, :].shape).astype(self.A_ij.dtype)
                        M = 2.0*self.noise_var * M + self.noise_var + 1.0
                        if not self.X_per_initialized:
                            self.X_per.append(_pin_mem(np.multiply(self.A_ij[i0:i1, :], M)))
                        else:
                            self.X_per[b] = np.multiply(self.A_ij[i0:i1, :], M)
                    else: # Row Axis
                        j0, j1 = b*self.batch_size, (b+1)*self.batch_size
                        M = np.random.random_sample(self.A_ij[:, j0:j1].shape).astype(self.A_ij.dtype)
                        M = 2.0*self.noise_var * M + self.noise_var + 1.0
                        if not self.X_per_initialized:
                            self.X_per.append(_pin_mem(np.multiply(self.A_ij[:, j0:j1], M)))
                        else:
                            self.X_per[b] = np.multiply(self.A_ij[:, j0:j1], M)

        else : #GPU Array
            for b in range(self.nBatch):
                if self.A_IS_SPARSE:
                    X = self.A_d[b]*1.0
                    M = cp.random.random_sample(self.A_d[b].data.shape, dtype=self.A_d[b].dtype)
                    M = 2.0*self.noise_var * M + self.noise_var + 1.0
                    X.data *= M
                    #self.X_per.append(X)
                    if not self.X_per_initialized:
                        self.X_per.append(X)
                    else:
                        self.X_per[b] = X
                else: # A in Dense
                    self.showMemStats(msg = " RandM() BEFORE X_per init  b%04d" %b)
                    MAXMEM = max(MAXMEM, self.MEMPOOL.total_bytes()*1.0)
                    if not self.X_per_initialized:
                        self.X_per.append(cp.random.random_sample(self.A_d[b].shape, dtype=self.A_d[b].dtype))
                        self.showMemStats(msg = " RandM() AFTER X_per init 1  b%04d" %b)
                        MAXMEM = max(MAXMEM, self.MEMPOOL.total_bytes()*1.0)
                    else:
                        self.X_per[b] = cp.random.random_sample(self.A_d[b].shape, dtype=self.A_d[b].dtype)
                        self.showMemStats(msg = " RandM() AFTER X_per set 1  b%04d" %b)
                        MAXMEM = max(MAXMEM, self.MEMPOOL.total_bytes()*1.0)
                    self.X_per[b] = 2.0*self.noise_var
                    self.X_per[b] += self.noise_var + 1.0
                    self.X_per[b] = cp.multiply(self.A_d[b], self.X_per[b])
                    self.showMemStats(msg = " RandM() X_per init OK  b%04d" %b)
                    if self.rank == 0: log(msg = f"[{amber('!')}][MEMPOOL]: randM(): MEM Peak = {blue(MAXMEM/self.GB)} GB", rank=self.rank, lrank=self.lrank)
                    self.MEMPOOL.free_all_blocks()

        if not self.X_per_initialized: self.X_per_initialized = True


    def init_factors(self):
        r"""
        Initializes NMF factor matrices W and H based on the selected initialization method.
        """
        # Random initialization
        if self.params.init == 'rand':
            if self.topo_dim=='2d':
                #raise Exception(f"[{red('!!')}] 2D topology is not yet supported on GPU")
                noSupport("2D topology")
            elif self.topo_dim in ['1d']:
                rng_h = np.random.RandomState(self.iter_seeds[self.k])
                rng_d = cp.random.RandomState(self.iter_seeds[self.k])
                if self.ROW_BATCHING_AXIS:
                    self.H_d     = cp.random.rand(self.k, self.grid_glob_n).astype(self.A_ij.dtype)               # H   [k, J]
                    self.H_is_cached =  True
                    if self.A_IS_BATCHED:  # W is distributed, H is replicate on Hostd
                        self.W_h = []
                        for n in range(self.nBatch):
                            self.W_h.append(_pin_mem(rng_h.rand(self.batch_size, self.k).astype(self.A_ij.dtype)))
                        self.W_is_cached = False
                    else:
                        self.W_d = rng_d.rand(self.grid_glob_m, self.k).astype(self.A_ij.dtype)  # W_h [m, k]
                        self.W_is_cached = True
                else: # COL_BATCHING_AXIS:
                    self.W_d = cp.random.rand(self.grid_glob_m, self.k).astype(self.A_ij.dtype)                   # W_d [I, k]
                    self.W_is_cached =  True
                    if self.A_IS_BATCHED:  # H is distributed, W is replicate on Hostd
                        self.H_h = []
                        for n in range(self.nBatch):
                            self.H_h.append(_pin_mem(rng_h.rand(self.k, self.batch_size).astype(self.A_ij.dtype)))
                        self.H_is_cached = False
                    else:
                        self.H_d = rng_d.rand(self.k, self.grid_glob_n).astype(self.A_ij.dtype)  # W_h [m, k]
                        self.H_is_cached = True
        # NNSVD-based initialization
        elif self.params.init in ['nnsvd','svd','singular']:
            if self.topo_dim == '1d':
                self.AA = []
                if self.A_IS_BATCHED:
                    for i in range(self.nGPUs):
                        self.AA.append(np.zeros(self.grid_L, self.grid_L))
                else:
                    for i in range(self.nGPUs):
                        self.AA.append(_zeros((self.grid_L, self.grid_L)))
                self.svdSoFar      = [[1,0,0]]
                self.W_i, self.H_j = self.nnsvd(flag=1, verbose=0)
            elif self.topo_dim == '2d':
                raise Exception('NNSVD init only available for 1D topology, please try with 1d topo.')
        else:
            # Raise an error if an unsupported initialization method is chosen
            raise Exception(f"[{red('!!')}] Only Random and nnsvd init is supported on GPU")




    def normalize_features(self):
        r"""
        Normalizes the NMF factor matrices W and H.
        """
        # If A is batched
        if self.A_IS_BATCHED: # W.I.P
            if self.COL_PARTITION: # W.I.P
                pass
            else: 
                Wall_norm = self.W_d.sum(axis=0, keepdims=True)
                if self.params.p_r != 1: self.nccl_comm.Allreduce(Wall_norm, Wall_norm, op=nccl.NCCL_SUM, stream=None)
                self.W_d/= Wall_norm+ self.eps
                self.H_h *=  cp.asnumpy(Wall_norm.T)  

        else: # If A is not batched
            Wall_norm = self.W_d.sum(axis=0, keepdims=True)
            if self.topo_dim == '2d':
                raise Exception(f"[{red('!!')}] 2D topology is not yet supported on GPU")
            elif self.topo_dim == '1d':
                if self.params.p_r != 1:
                    self.nccl_comm.Allreduce(Wall_norm, Wall_norm, op=nccl.NCCL_SUM, stream=None)
            else:
                raise Exception(f"[{red('!!')}] Topology {self.topo_dim}is not yet supported")
            self.W_d /= Wall_norm+ self.eps
            self.H_d *= Wall_norm.T


    def relative_err(self):
        r"""
        Computes the relative reconstruction error of the NMF decomposition.
        """

        # Check topology and raise an error for unsupported topologies
        if self.topo_dim == '2d': raise Exception(f"[{red('!!')}] 2D topology is not yet supported on GPU")
        err = 0.0
        # Get the max available memory in bytes
        MAXMEM = self.MEMPOOL.total_bytes()*1.0

        # Compute the error for batched CPU arrays
        if self.A_IS_BATCHED: #CPU Array
            for b in range(self.nBatch):
                if self.ROW_BATCHING_AXIS: # ROW AXIS
                    self.H_h = cp.asnumpy(self.H_d) # Download H
                    if self.A_IS_SPARSE:
                        err += np.square( self.X_per[b].toarray() - (self.W_h[b] @ self.H_h) ).sum()
                    else:
                        err += np.square( self.X_per[b] - (self.W_h[b] @ self.H_h) ).sum()
                else:                  # COL AXIS
                    self.W_h = cp.asnumpy(self.W_d) # Downlod W
                    if self.A_IS_SPARSE:
                        #err += np.square( self.X_per[b].toarray() - (self.W_h @ self.H_h[b]) ).sum()
                        err += 0.0
                    else:
                        err += np.square( self.X_per[b] - (self.W_h @ self.H_h[b]) ).sum()
            self.glob_norm_A = self.dist_norm(self.A_ij)
        else:     # Compute the error for GPU arrays
            #self.showMemStats(msg = " relative_err() STARTING BATCHES")
            MAXMEM = max(MAXMEM, self.MEMPOOL.total_bytes()*1.0)
            for b in range(self.nBatch):
                if self.ROW_BATCHING_AXIS: # Row Partition
                    i0, i1     = b*self.grid_I, (b+1)*self.grid_I
                    if self.A_IS_SPARSE:
                        err += cp.square( self.A_d[b].toarray() - (_matmul(self.W_d[i0:i1], self.H_d)) ).sum()
                    else:

                        err += cp.square( self.A_d[b] - (_matmul(self.W_d[i0:i1], self.H_d)) ).sum()
                        MAXMEM = max(MAXMEM, self.MEMPOOL.total_bytes()*1.0)
                        self.MEMPOOL.free_all_blocks()

                else:                  # Col Partition
                    j0, j1     = b*self.grid_J, (b+1)*self.grid_J
                    if self.A_IS_SPARSE:
                        err += cp.square( self.A_d[b].toarray() - (_matmul(self.W_d, self.H_d[:, j0:j1])) ).sum()
                    else:
                        err += cp.square( self.A_d[b] - (_matmul(self.W_d, self.H_d[:, j0:j1])) ).sum()
            err = err.get()
            self.glob_norm_A = self.dist_norm(self.A_d)
            #self.glob_norm_A = self.dist_norm(self.X_per)
            #del err
        err= self.sub_comm.allreduce(err)
        self.glob_norm_err = np.sqrt(err)
        self.recon_err = self.glob_norm_err / self.glob_norm_A
        if self.rank == 0: log(msg = f"[{amber('!')}][MEMPOOL]: relative_err(): MEM Peak = {blue(MAXMEM/self.GB)} GB", rank=self.rank, lrank=self.lrank) 

    def dist_norm(self, X, proc=-1, norm='fro', axis=None):
        r"""Computes the distributed norm of an array.

        Args:
        - X (list/array): The input array or list of arrays.
        - proc (int, optional): Processor ID. Default is -1.
        - norm (str, optional): Type of matrix norm. Default is 'fro' (Frobenius norm).
        - axis (optional): Axis for norm computation. Default is None.

        Returns:
        - float: The computed distributed norm.
        """

        # If X is a list, this is a batched computation.
        if type(X) in [list]: # BACHED
            array_type = self.getArrayType(X[0]) 
            N  = len(X)
            err = 0.0
            try:
                device, density = array_type.split('_')
            except:
                raise Exception('[!][dist_norm()] UNABLE TO IDENTIFY ARRAY OBJECT TYPE {}!!'.format(array_type))
            if device in ['CPU']:
                if density in ['DENSE']:
                    for n in range(N): err += np.square(X[n]).sum()
                elif density in ['SPARSE']:
                    for n in range(N): err += np.square(X[n].data).sum()
                else:
                    raise Exception('[!][dist_norm()] UNABLE TO IDENTIFY ARRAY OBJECT TYPE {}!!'.format(array_type))
            elif device in ['GPU']:
                if density in ['DENSE']:
                    for n in range(N): err += cp.square(X[n]).sum()
                elif density in ['SPARSE']:
                    for n in range(N): err += cp.square(X[n].data).sum()
                else:
                    raise Exception('[!][dist_norm()] UNABLE TO IDENTIFY ARRAY OBJECT TYPE {}!!'.format(array_type))
                err = err.get()
                
            else:
                raise Exception('[!][dist_norm()] UNABLE TO IDENTIFY ARRAY OBJECT TYPE {}!!'.format(array_type))
        else:                # If X is not a list, this is a non-batched computation.
            array_type = self.getArrayType(X)
            try:
                device, density = array_type.split('_')
            except:
                raise Exception('[!][dist_norm()] UNABLE TO IDENTIFY ARRAY OBJECT TYPE {}!!'.format(array_type))
            if device in ['CPU']:
                if density in ['DENSE']:
                    err=np.square(X).sum()
                elif density in ['SPARSE']:
                    err=np.square(X.data).sum()
            elif device in ['GPU']:
                if density in ['DENSE']:
                    err = cp.square(X).sum()
                elif density in ['SPARSE']:
                    err = cp.square(X.data).sum()
                else:
                    raise Exception('[!][dist_norm()] UNABLE TO IDENTIFY ARRAY OBJECT TYPE {}!!'.format(array_type))
                err = err.get()
            else:
                raise Exception('[!][dist_norm()] UNABLE TO IDENTIFY ARRAY OBJECT TYPE {}!!'.format(array_type))

        err = self.sub_comm.allreduce(err)
        return np.sqrt(err)


    def fit(self, factors=None, save_factors=False, rtrn=['W','H']):
        r"""Fits the NMF model based on provided parameters.

        Args:
        - factors (list, optional): Initial factor matrices. Default is None.
        - save_factors (bool, optional): Flag to decide whether to save factor matrices. Default is False.
        - rtrn (list, optional): List of matrix names to return. Default is ['W','H'].

        Returns:
        - tuple: Factor matrices and meta information.
        """
        if self.params.norm.upper() == 'FRO':
            if self.params.method.upper() == 'MU':
                self.Fro_MU_update(factors=factors, save_factors=save_factors, rtrn=rtrn)
            else:
                raise Exception('Not a valid method: Choose (mu)')
        elif self.params.norm.upper() == 'KL':
            if self.params.method.upper() == 'MU':
                self.KL_MU_update(factors=factors, save_factors=save_factors, rtrn=rtrn)
            else:
                raise Exception('Not a valid method: Choose (mu)')
        else:
            raise Exception('Not a valid norm: Choose (fro/kl)')
        #return self.W_i, self.H_j
        metha        = {}
        metha['dt']  = self.dt
        metha['err'] = self.recon_err
        if self.A_IS_BATCHED:
            if self.ROW_BATCHING_AXIS:
                return self.W_h, cp.asnumpy(self.H_d), metha
            else:
                return cp.asnumpy(self.W_d), self.H_h, metha
            #return self.W_h, self.H_h, self.recon_err
        else:
            return cp.asnumpy(self.W_d), cp.asnumpy(self.H_d), metha #self.recon_err.get()


    def Fro_MU_update(self, factors=None, save_factors=False, rtrn=['W','H']):
        r"""Performs updates for NMF using the Frobenius norm and MU optimization method.

        Args:
        - factors (list, optional): Initial factor matrices. Default is None.
        - save_factors (bool, optional): Flag to decide whether to save factor matrices. Default is False.
        - rtrn (list, optional): List of matrix names to return. Default is ['W','H'].

        Returns:
        - None
        """
        # Memory management and initialization steps.
        self.showMemStats(msg = " Fro_MU_update() 0")
        self.MEMPOOL.free_all_blocks()
        #self.showMemStats(msg = " Fro_MU_update() MEMPOOL Freed OK")
        self.PINNEDMEMPOOL.free_all_blocks()
        # If the matrix is batched, perform the updates using the appropriate method based on partitioning axis.
        if self.A_IS_BATCHED: # BATCHED CPU -> GPU
            if   self.COL_BATCHING_AXIS: 
                self.FroNMF_1D_row(factors=factors, save_factors=save_factors, rtrn=rtrn)
            elif self.ROW_BATCHING_AXIS:
                #self.FroNMF_1D_col(factors=factors, save_factors=save_factors, rtrn=rtrn)
                self.FroNMF_1D_row_batched(factors=factors, save_factors=save_factors, rtrn=rtrn)
            else:
                raise Exception(f"[{red('!!')}] Grid Partition UNDEFINED!! ")
        else:                 # LOCAL BATCHING on GPU Mem
            if  self.COL_BATCHING_AXIS:
                self.FroNMF_1D_col_partion(factors=factors, save_factors=save_factors, rtrn=rtrn)
            elif self.ROW_BATCHING_AXIS:
                self.FroNMF_1D_row_partion(factors=factors, save_factors=save_factors, rtrn=rtrn)
            else:
                raise Exception(f"[{red('!!')}] Grid Partition UNDEFINED!! ")
    

    def FroNMF_1D_col_partion(self,factors=None, save_factors=False, rtrn=False): #W.I.P
        r"""Performs NMF on 1D column partitioned data using the Frobenius norm.

        Args:
        - factors (list, optional): Initial factor matrices. Default is None.
        - save_factors (bool, optional): Flag to decide whether to save factor matrices. Default is False.
        - rtrn (bool, optional): Flag to decide whether to return factor matrices. Default is False.

        Returns:
        - None
        """
        #[1] Initialize perturbed cofactors
        self.init_factors()
        #[2] Initialiaze different accumulators
        self.events['nmf_start'].record()
        self.dt['W_up'], self.dt['H_up'], self.dt['allRed_XHT'], self.dt['allRed_WHHT'] = 0.0, 0.0, 0.0, 0.0
        for i in range(self.params.itr):
            #[3] Initialize iteration Accumulators
            dt_w, dt_h, dt_reduce1, dt_reduce2   = 0.0, 0.0, 0.0, 0.0
            WT   =  self.W_d.T                                                   # [m, k].T        -> WT   [k, m]
            WTW  =  _matmul(WT, self.W_d)                                        # [k, m] @ [m, k] -> WTW  [k, k]
            WHHT = _zeros((self.grid_glob_m, self.k), dtype=self.A_ij.dtype)     # WHHT [m, k] # ACCUMULATOR for W@H@H.T
            XHT = _zeros((self.grid_glob_m, self.k), dtype=self.A_ij.dtype)      # XHT  [m, k] # ACCUMULATOR for X@H.T
            #[4] Strart local batching
            for b in range(self.nBatch):
                j0, j1 = b*self.batch_size, (b+1)*self.batch_size
                #[5]////////////////////////////  (H update)> ////////////////////////////////////////////////////////
                self.events['start'].record()
                WTWH = _matmul(WTW, self.H_d[:, j0:j1]) + self.eps           # [k, k] @ [k, b] -> WTWH [k, b]
                if self.A_IS_SPARSE:
                    WTX  = self.MM(self.X_per[p], self.W_d, transpA=True).T  # ([m, b].T @ [m, k]).T -> [b, k].T ->  WTX [k, b]
                else:
                    WTX  = _matmul(WT, self.X_per[b])                        # [k, m] @ [m, b] -> XHT  [k, b]
                WTX = _multiply(WTX, self.H_d[:, j0:j1])                     # [k, b] * [k, b] -> HWTX [k, b]
                self.H_d[:, j0:j1] = _divide(WTX, WTWH)                      # [k, b] / [k, b] -> H    [k, b]
                self.events['end'].record()
                self.events['end'].synchronize()
                #del  WTWH, WTX
                #self.MEMPOOL.free_all_blocks()
                dt_h += cp.cuda.get_elapsed_time(self.events['start'], self.events['end'])  # [ms]
                #[6]////////////////////////////  (W update)> ////////////////////////////////////////////////////////
                if self.params.W_update:
                    self.events['start'].record()
                    HT   = self.H_d[:, j0:j1].T                                    # [k, b].T        -> HT   [b, k]
                    HHT  = _matmul(self.H_d[:, j0:j1], HT)                         # [k, b] @ [b, k] -> HTH  [k, k]
                    WHHT += _matmul(self.W_d, HHT)                                 # [m, k] @ [k, k] -> WHHT [m, k]
                    if self.A_IS_SPARSE:
                        XHT += self.MM(self.X_per[p], HT, transpA=False)           # [b, n].T @ [b, k] -> [n, k].T -> [k, n]
                    else:
                        XHT += _matmul(self.self.X_per[b], HT)                     # [m, b] @ [b, k] -> xht  [m, k]
                    self.events['end'].record()
                    self.events['end'].synchronize()
                    #del HT, HHT 
                    #self.MEMPOOL.free_all_blocks()
                    dt_w += cp.cuda.get_elapsed_time(self.events['start'], self.events['end'])  # [ms]
            #[7] Aggregate Global WHHT and XHT
            self.events['start'].record()
            if self.nGPUs > 1:
                self.events['reduce_s'].record()
                nccl_comm.Allreduce(WHHT, WHHT, op=nccl.NCCL_SUM, stream=None)
                self.events['reduce_e'].record()
                self.events['reduce_e'].synchronize()
                dt_reduce1 += cp.cuda.get_elapsed_time(self.events['reduce_s'], self.events['reduce_e'])
                self.events['reduce_s'].record()
                nccl_comm.Allreduce(XHT, XHT, op=nccl.NCCL_SUM, stream=None)
                self.events['reduce_e'].record()
                self.events['reduce_e'].synchronize()
                dt_reduce2 += cp.cuda.get_elapsed_time(self.events['reduce_s'], self.events['reduce_e'])
            #[8] Finalize W-Update
            WHHT     =  WHHT + self.eps
            XHT      = _multiply(self.W_d, XHT)                                # [m, k] * [m, k] ->      [m, k]
            self.W_d = _divide(XHT, WHHT)                                      # [m, k] / [m, k] ->      [m, k]
            self.events['end'].record()
            self.events['end'].synchronize()
            dt_w += cp.cuda.get_elapsed_time(self.events['start'], self.events['end'])  # [ms]
            #[9] Accumulate iteration performance metric
            self.dt['W_up']        += dt_w*1.0
            self.dt['H_up']        += dt_h*1.0
            self.dt['allRed_WHHT'] += dt_reduce1*1.0
            self.dt['allRed_XHT']  += dt_reduce2*1.0
            #[10]
            if i % 10 == 0:
                #if self.A_is_Large:
                #    self.H_h = np.maximum(self.H_h, self.eps)
                #else:
                #    self.H_d = cp.maximum(self.H_d, self.eps)
                self.H_d = cp.maximum(self.H_d, self.eps)
                self.W_d = cp.maximum(self.W_d, self.eps)

        #[10] Compute Average  performance metric
        self.dt['W_up']        /= float(self.params.itr)
        self.dt['H_up']        /= float(self.params.itr)
        self.dt['allRed_XHT']  /= float(self.params.itr)
        self.dt['allRed_WHHT'] /= float(self.params.itr)
        self.events['nmf_end'].record()
        self.events['nmf_end'].synchronize()
        self.dt['NMF'] = cp.cuda.get_elapsed_time(self.events['nmf_start'], self.events['nmf_end'])  # [ms]
        #[11] Calculate relative error
        self.relative_err()



    def FroNMF_1D_row_partion(self,factors=None, save_factors=False, rtrn=False):
        r"""
        This method performs Frobenius Norm Non-negative Matrix Factorization (FroNMF)
        on a 1D row partitioned matrix.

        Parameters:
        - factors: Initial guess for the factors (Optional)
        - save_factors: Boolean, whether to save the computed factors or not (Default is False)
        - rtrn: Boolean, whether to return the factors (Default is False)
        """

        # [1] Monitor initial memory usage
        self.showMemStats(msg = " FroNMF_1D_row_partion() 0")

        # [2] Initialization of perturbed cofactors
        self.init_factors()
        #self.showMemStats(msg = " FroNMF_1D_row_partion() init_factors OK")
        # [3] Initialize various accumulators for performance measurement and computation

        self.events['nmf_start'].record()
        self.dt['W_up'], self.dt['H_up'], self.dt['allRed_WTX'], self.dt['allRed_WTWH'] = 0.0, 0.0, 0.0, 0.0

        # Main iterative loop for NMF computation
        for i in range(self.params.itr):
            # Initialize accumulators for this iteration
            dt_w, dt_h, dt_reduce1, dt_reduce2   = 0.0, 0.0, 0.0, 0.0

            # [4] Setup for W matrix update
            if self.params.W_update:
                HT   =  self.H_d.T                                             # HT [n, k]
                HHT  =  _matmul(self.H_d, HT)                                  # [k, n] @ [n, k] -> HHT  [k, k]

            # Initialize accumulators for batch computations
            WTWH = _zeros((self.k, self.grid_glob_n), dtype=self.A_ij.dtype)   # WTWH [k, n] # ACCUMULATOR for WT@W@H
            WTX  = _zeros((self.k, self.grid_glob_n), dtype=self.A_ij.dtype)   # WTX  [k, n] # ACCUMULATOR for WT@A

            # [5] Batching loop for local computations
            for b in range(self.nBatch):
                # Determine current batch range
                i0, i1 = b*self.batch_size, (b+1)*self.batch_size
                #self.showMemStats(msg = " FroNMF_1D_row_partion() 2 b%04d" %b)
                #[6]////////////////////////////  (W update)> ////////////////////////////////////////////////////////
                if self.params.W_update:
                    self.events['start'].record()
                    WHHT = _matmul(self.W_d[i0:i1,:], HHT) + self.eps          # [b, k] @ [k, k] -> WHHT [b, k]
                    #self.showMemStats(msg = " FroNMF_1D_row_partion() 3 b%04d" %b)
                    if self.A_IS_SPARSE:
                        XHT  = self.MM(self.X_per[b], HT, transpA=False)       # [b, n] @ [n, k] -> XHT  [b, k]
                    else:
                        XHT  = _matmul(self.X_per[b], HT)                      # [b, n] @ [n, k] -> XHT  [b, k]
                    #self.showMemStats(msg = " FroNMF_1D_row_partion() 4 b%04d" %b)
                    XHT  = _multiply(self.W_d[i0:i1,:], XHT)                   # [b, k] * [b, k] -> WXHT [b, k]
                    #self.showMemStats(msg = " FroNMF_1D_row_partion() 5 b%04d" %b)
                    self.W_d[i0:i1,:] = _divide(XHT, WHHT)                     # [b, k] / [b, k] -> W    [b, k]
                    #self.showMemStats(msg = " FroNMF_1D_row_partion() W-up  b%04d OK" %b)
                    self.events['end'].record()
                    self.events['end'].synchronize()
                    del  WHHT, XHT
                    self.MEMPOOL.free_all_blocks()
                    dt_w += cp.cuda.get_elapsed_time(self.events['start'], self.events['end'])  # [ms]
                #[7]////////////////////////////  (H update)> ////////////////////////////////////////////////////////
                #self.showMemStats(msg = " FroNMF_1D_row_partion() 7")
                self.events['start'].record()
                WT   = self.W_d[i0:i1,:].T                                     # [b, k].T        -> WT   [k, b]
                #self.showMemStats(msg = " FroNMF_1D_row_partion() 8")
                WTW  = _matmul(WT, self.W_d[i0:i1,:])                          # [k, b] @ [b, k] -> WTW  [k, k]
                #self.showMemStats(msg = " FroNMF_1D_row_partion() 9")
                #print(f"[!!!][iter %04d | batch %04d] shape(WTW)={WTW.shape} || shape(H_d) = {self.H_d.shape}" %(i, b))
                WTWH += _matmul(WTW, self.H_d)                                 # [k, k] @ [k, n] -> WTWH [k, n]
                #self.showMemStats(msg = " FroNMF_1D_row_partion() 10 ")
                if self.A_IS_SPARSE:
                    WTX  += self.MM(self.X_per[b], self.W_d[i0:i1,:], transpA=True).T # [b, n].T @ [b, k] -> [n, k].T -> [k, n]
                else:
                    WTX  += _matmul(WT, self.X_per[b])                    # [k, b] @ [b, n] -> WTX  [k, n]
                #self.showMemStats(msg = " FroNMF_1D_row_partion() H-up B%04d OK" %b)
                self.events['end'].record()
                self.events['end'].synchronize()
                # Free memory from intermediate variables
                del WT, WTW 
                self.MEMPOOL.free_all_blocks()
                dt_h += cp.cuda.get_elapsed_time(self.events['start'], self.events['end'])  # [ms]

            # Free memory from intermediate variable
            del HHT
            self.MEMPOOL.free_all_blocks()
            # [8] Aggregate global WTWH and WTX values (used for multi-GPU setup)
            self.events['start'].record()
            if self.nGPUs > 1:
                # Perform Allreduce operation on WTWH and WTX
                # This is to aggregate results from all GPUs

                self.events['reduce_s'].record()
                self.nccl_comm.Allreduce(WTWH, WTWH, op=nccl.NCCL_SUM, stream=None)
                self.events['reduce_e'].record()
                self.events['reduce_e'].synchronize()
                dt_reduce1 += cp.cuda.get_elapsed_time(self.events['reduce_s'], self.events['reduce_e'])
                self.events['reduce_s'].record()
                self.nccl_comm.Allreduce(WTX, WTX, op=nccl.NCCL_SUM, stream=None)
                self.events['reduce_e'].record()
                self.events['reduce_e'].synchronize()
                dt_reduce2 += cp.cuda.get_elapsed_time(self.events['reduce_s'], self.events['reduce_e'])
            # [9] Update H with global results
            WTWH     =  WTWH + self.eps
            #self.showMemStats(msg = " FroNMF_1D_row_partion() 12")
            WTX      = _multiply(self.H_d, WTX)                                # [k, n] * [k, n] ->      [k, n]
            #self.showMemStats(msg = " FroNMF_1D_row_partion() 13")
            self.H_d = _divide(WTX, WTWH)                                      # [k, n] / [k, n] ->      [k, n]
            #self.showMemStats(msg = " FroNMF_1D_row_partion() NMF OK")
            self.events['end'].record()
            self.events['end'].synchronize()
            dt_h += cp.cuda.get_elapsed_time(self.events['start'], self.events['end'])  # [ms]
            # Accumulate iteration performance metric
            self.dt['W_up']        += dt_w*1.0
            self.dt['H_up']        += dt_h*1.0
            self.dt['allRed_WTX']  += dt_reduce1*1.0
            self.dt['allRed_WTWH'] += dt_reduce2*1.0
            #[10] Clipping to ensure non-negativity after every 10 iterations
            if i % 10 == 0:
                #if self.A_is_Large:
                #    self.H_h = np.maximum(self.H_h, self.eps)
                #else:
                #    self.H_d = cp.maximum(self.H_d, self.eps)
                self.H_d = cp.maximum(self.H_d, self.eps)
                self.W_d = cp.maximum(self.W_d, self.eps)
        self.showMemStats(msg = " FroNMF_1D_row_partion() NMF OK")
        # [11] Compute Average Performance metrics over all iterations
        del WTWH,  WTX
        self.MEMPOOL.free_all_blocks()
        self.showMemStats(msg = " FroNMF_1D_row_partion() ALL DELETIONS DONE")
        self.dt['W_up']        /= float(self.params.itr)
        self.dt['H_up']        /= float(self.params.itr)
        self.dt['allRed_WTX']  /= float(self.params.itr)
        self.dt['allRed_WTWH'] /= float(self.params.itr)
        self.events['nmf_end'].record()
        self.events['nmf_end'].synchronize()
        self.dt['NMF'] = cp.cuda.get_elapsed_time(self.events['nmf_start'], self.events['nmf_end'])  # [ms]
        # [12] Calculate relative error
        self.showMemStats(msg = " FroNMF_1D_row_partion() BEFORE ERROR Eval")
        if self.A_IS_SPARSE:
            self.recon_err = 0.0
        else:
            self.relative_err()
        self.showMemStats(msg = " FroNMF_1D_row_partion() AFTER  ERROR Eval")
        # Free all memory blocks
        self.MEMPOOL.free_all_blocks()
        self.showMemStats(msg = " FroNMF_1D_row_partion() LAST CALL MEM POOL FReed OK")



    def FroNMF_1D_row_batched(self,factors=None, save_factors=False, rtrn=False): #W.I.P
        r"""
        Computes the Frobenius Norm Non-negative Matrix Factorization (FroNMF) for the 1D row of the data matrix
        using a batched approach.

        Args:
        - factors: Not used in this function but could represent the initial factors for matrix factorization.
        - save_factors: A boolean flag to determine if factors should be saved.
        - rtrn: A boolean flag to determine if the function should return some values (not implemented yet).

        Returns:
        - None (but updates internal attributes of the object and potentially could return some values based on `rtrn` flag).
        """
        #[1] Initialize perturbed cofactors
        self.init_factors()
        #[2] Initialize accumulators and record the start event for NMF
        self.events['nmf_start'].record()
        self.dt['W_up'], self.dt['H_up'], self.dt['allRed_WTX'], self.dt['allRed_WTWH'] = 0.0, 0.0, 0.0, 0.0
        self.dt['H2D_A'], self.dt['H2D_W'], self.dt['D2H_W'] = 0.0, 0.0, 0.0
        SQ = self.get_managed_stream_queue()                             # Get Managed stream Queue

        # Loop for the number of iterations (NMF optimization steps)
        for i in range(self.params.itr):
            #[3] Initialize iteration Accumulators
            dt_w, dt_h, dt_reduce1, dt_reduce2   = 0.0, 0.0, 0.0, 0.0
            if self.params.W_update:
                HT   =  self.H_d.T                                             # HT [n, k]
                HHT  =  _matmul(self.H_d, HT)                                  # [k, n] @ [n, k] -> HHT  [k, k]
            #WTWH = _zeros((self.k, self.grid_glob_n), dtype=self.A_ij.dtype)   # WTWH [k, n] # ACCUMULATOR for WT@W@H
            ##WTX  = _zeros((self.k, self.grid_glob_n), dtype=self.A_ij.dtype)   # WTX  [k, n] # ACCUMULATOR for WT@A
            WTX, WTWH = [], []
            H2D_A, H2D_W, D2H_W, W_up, H_up, AR_1, AR_2 = [], [], [], [], [], [], []
            #[4] Start processing data in batches
            for b in range(self.nBatch):
                st_key = SQ['Queue'].get()                                              # Get stream from the queue. If queue is empty, wait until a stream is available.
                stream = SQ['READY'][st_key]
                i0, i1 = b*self.batch_size, (b+1)*self.batch_size
                #strm_idx  = b % self.MAXSTREAMS
                #strm_name = self.stream[strm_idx]
                #stream    = self.STREAMS[strm_name]
                with stream:
                    Q_idx = b % self.batchQeueSize
                    #stream.wait_event(self.FREE_STREAM[strm_name])
                    self.events['h2d_s'].record()
                    self.W_d[Q_idx].set(self.W_h[b])
                    self.events['h2d_e'].record()
                    self.events['h2d_e'].synchronize()
                    H2D_W.append( cp.cuda.get_elapsed_time(self.events['h2d_s'], self.events['h2d_e']) )  # [ms]
                    self.events['h2d_s'].record()
                    if self.A_IS_SPARSE: # CopyH2D_Async Sparse array
                        #print(f"[!!] [itr %04d][b %02d] type(X_per[b])= {self.X_per[b].dtype}" %(i,b))
                        self.X_d[Q_idx].set(self.X_per[b].astype(self.A_ij.dtype))
                        self.buff_idx[Q_idx].set(self.X_idx[b])
                        self.buff_ptr[Q_idx].set(self.X_ptr[b])
                    else:
                        self.X_d[Q_idx].set(self.X_per[b]) # CopyH2D_Async
                    self.events['h2d_e'].record()
                    self.events['h2d_e'].synchronize()
                    #print(f"[!!] [itr %04d][b %02d] H2D_A = {cp.cuda.get_elapsed_time(self.events['h2d_s'], self.events['h2d_e'])}" %(i,b))
                    H2D_A.append( cp.cuda.get_elapsed_time(self.events['h2d_s'], self.events['h2d_e']) )  # [ms]

                    #[5]////////////////////////////  (W update)> ////////////////////////////////////////////////////////
                    if self.params.W_update:
                        self.events['start'].record()
                        WHHT = _matmul(self.W_d[Q_idx], HHT) + self.eps          # [b, k] @ [k, k] -> WHHT [b, k]
                        if self.A_IS_SPARSE:
                            #XHT  = self.MM(self.X_per[p], HT, transpA=False)       # [b, n] @ [n, k] -> XHT  [b, k] 
                            #print(f"[!!] BS = {self.batch_size} || vect_size_dat = {self.sparse_vect_size['dat'][b]}, vect_size_idx = {self.sparse_vect_size['idx'][b]}, vect_size_ptr = {self.sparse_vect_size['ptr'][b]},")
                            X_d = cupyx.scipy.sparse.csr_matrix((self.X_d[Q_idx][:self.sparse_vect_size['dat'][b]],
                                    self.buff_idx[Q_idx][:self.sparse_vect_size['idx'][b]],
                                    self.buff_ptr[Q_idx][:self.sparse_vect_size['ptr'][b]]), shape=(self.batch_size, self.grid_loc_n))
                            #print(f"[!!] shape(X_d) ={X_d.shape} ||  shape(HT) = {HT.shape} || BS = {self.batch_size} || vect_size_dat = {self.sparse_vect_size['dat'][b]}")
                            #XHT  = self.MM(X_d, HT, transpA=False)                 # [b, n] @ [n, k] -> XHT  [s, k]
                            XHT   = X_d @ HT
                        else:
                            #XHT  = _matmul(self.X_per[b], HT)                      # [b, n] @ [n, k] -> XHT  [b, k]
                            XHT  = self.MM(self.X_d[Q_idx], HT)                     # [b, n] @ [n, k] -> XHT  [s, k]
                        #XHT  = _multiply(self.W_d[i0:i1,:], XHT)                   # [b, k] * [b, k] -> WXHT [b, k]
                        #self.W_d[i0:i1,:] = _divide(XHT, WHHT)                     # [b, k] / [b, k] -> W    [b, k]
                        self.W_d[Q_idx]   = _divide( _multiply(self.W_d[Q_idx], XHT), _matmul(self.W_d[Q_idx],HHT)+self.params.eps) # [bs,k] <- ([bs,k]*[bs,k]) / ([bs,k]@[k, k] + epsi)
                        #WT  = self.W_d[Q_idx].T                                     # [k, b]
                        self.events['d2h_s'].record()
                        self.W_d[Q_idx].get(out=self.W_h[b])                   # CopyD2H_Async W_d -> W_h
                        self.events['d2h_e'].record()
                        self.events['d2h_e'].synchronize()
                        D2H_W.append( cp.cuda.get_elapsed_time(self.events['d2h_s'], self.events['d2h_e']) )  # [ms]
                        self.events['end'].record()
                        self.events['end'].synchronize()
                        #del  WHHT, XHT
                        #self.MEMPOOL.free_all_blocks()
                        W_up.append(cp.cuda.get_elapsed_time(self.events['start'], self.events['end']) )  # [ms]
                    #[6]////////////////////////////  (H update)> ////////////////////////////////////////////////////////
                    self.events['start'].record()
                    #WT   = self.W_d[i0:i1,:].T                                    # [b, k].T        -> WT   [k, b]
                    WT  = self.W_d[Q_idx].T                                        # [b, k].T        -> WT   [k, b]
                    #WTW  = _matmul(WT, self.W_d[i0:i1,:])                          # [k, b] @ [b, k] -> WTW  [k, k]
                    WTW  = _matmul(WT, self.W_d[Q_idx])                          # [k, b] @ [b, k] -> WTW  [k, k]
                    #print(f"[!!!][iter %04d | batch %04d] shape(WTW)={WTW.shape} || shape(H_d) = {self.H_d.shape}" %(i, b))
                    #WTWH += _matmul(WTW, self.H_d)                                 # [k, k] @ [k, n] -> WTWH [k, n]
                    WTWH.append(_matmul(WTW, self.H_d))                            # [k, k] @ [k, n] -> WTWH [k, n]
                    if self.A_IS_SPARSE:
                        #WTX  += self.MM(X_d, self.W_d[Q_idx], transpA=True).T # [b, n].T @ [b, k] -> [n, k].T -> [k, n]
                        WTX.append(self.MM(X_d, self.W_d[Q_idx], transpA=True).T) # [b, n].T @ [b, k] -> [n, k].T -> [k, n]
                    else:
                        #WTX  += _matmul(WT, self.X_d[Q_idx])                    # [k, b] @ [b, n] -> WTX  [k, n]
                        WTX.append(_matmul(WT, self.X_d[Q_idx]))                    # [k, b] @ [b, n] -> WTX  [k, n]
                    self.events['end'].record()
                    self.events['end'].synchronize()
                    #del WT, WTW 
                    #self.MEMPOOL.free_all_blocks()
                    H_up.append(cp.cuda.get_elapsed_time(self.events['start'], self.events['end'])) # [ms]
                    completed_event = stream.record()
                    SQ['COMPLETED'].append(completed_event)
                    SQ['Queue'].put(st_key)
            # Make sure all streams are done
            for b in range(len(SQ['COMPLETED'])):
                SQ['REDUCE'].wait_event(SQ['COMPLETED'][b])
            cp.cuda.Device().synchronize()

            # Local reduces
            WTWH, WTX = sum(WTWH), sum(WTX)
            #device.synchronize() # Wait for Local reduce to finish
            #[7] GLOBAL REDUCE: Aggregate Global WTWH and WTX
            self.events['start'].record()
            if self.nGPUs > 1:
                self.events['reduce_s'].record()
                self.nccl_comm.Allreduce(WTWH, WTWH, op=nccl.NCCL_SUM, stream=None)
                self.events['reduce_e'].record()
                self.events['reduce_e'].synchronize()
                dt_reduce1 += cp.cuda.get_elapsed_time(self.events['reduce_s'], self.events['reduce_e'])
                self.events['reduce_s'].record()
                self.nccl_comm.Allreduce(WTX, WTX, op=nccl.NCCL_SUM, stream=None)
                self.events['reduce_e'].record()
                self.events['reduce_e'].synchronize()
                dt_reduce2 += cp.cuda.get_elapsed_time(self.events['reduce_s'], self.events['reduce_e'])
            #[8] Perform H-Update
            WTWH     =  WTWH + self.eps
            WTX      = _multiply(self.H_d, WTX)                                # [k, n] * [k, n] ->      [k, n]
            self.H_d = _divide(WTX, WTWH)                                      # [k, n] / [k, n] ->      [k, n]
            self.events['end'].record()
            self.events['end'].synchronize()
            dt_h += cp.cuda.get_elapsed_time(self.events['start'], self.events['end'])  # [ms]
            #[9] Accumulate iteration performance metric
            #print(f"[!!] [itr %04d]  H2D_A = {sum(H2D_A)}" %i)
            self.dt['W_up']        += dt_w*1.0
            self.dt['H_up']        += dt_h*1.0
            self.dt['allRed_WTX']  += dt_reduce1*1.0
            self.dt['allRed_WTWH'] += dt_reduce2*1.0
            self.dt['H2D_A']       += sum(H2D_A)
            self.dt['H2D_W']       += sum(H2D_W)
            self.dt['D2H_W']       += sum(D2H_W)
            #[10]
            #if i % 2000 == 0:
            if False:
                self.H_d = cp.maximum(self.H_d, self.eps)
                self.W_d = cp.maximum(self.W_d, self.eps)
            #self.relative_err()
            #print(f"[+] [iter%08d] err = {self.recon_err}" %i)

        #[10] Compute Average  performance metric
        self.dt['W_up']        /= float(self.params.itr)
        self.dt['H_up']        /= float(self.params.itr)
        self.dt['allRed_WTX']  /= float(self.params.itr)
        self.dt['allRed_WTWH'] /= float(self.params.itr)
        self.dt['H2D_A']       /= float(self.params.itr)
        self.dt['H2D_W']       /= float(self.params.itr)
        self.dt['D2H_W']       /= float(self.params.itr)
        print(f"[!!!!] H2D(A) = {self.dt['H2D_A']} | H2D(W) = {self.dt['H2D_W']} | D2H(W) = {self.dt['D2H_W']}")
        if False: #i % 10 == 0:
            if self.A_IS_BATCHED:
                self.H_h = np.maximum(self.H_h, self.eps)
            else:
                self.H_d = cp.maximum(self.H_d, self.eps)
            self.W_d = cp.maximum(self.W_d, self.eps)
        if i == self.params.itr - 1:
            self.events['nmf_end'].record()
            self.events['nmf_end'].synchronize()
            self.dt['NMF'] = cp.cuda.get_elapsed_time(self.events['nmf_start'], self.events['nmf_end'])
            #self.W_d, self.H_d = self.normalize_features(self.W_d, self.H_d)
            #self.normalize_features()
            if self.A_IS_SPARSE:
                self.recon_err = 0.0
            else:
                self.relative_err()
                #self.recon_err = 0.0
        else:
            print(f"[!] NOT OOOOOOOOK   i = {i} | self.params.itr = {self.params.itr }")
        del SQ




    def FroNMF_1D_col(self,factors=None, save_factors=False, rtrn=False):
        r"""
        Performs Frobenius Non-negative Matrix Factorization (NMF) using
        1D column-based partitioning with GPU acceleration.

        Parameters:
        - factors : (Optional) Initial values for matrix factors.
        - save_factors : (Optional) If True, saves factorized matrices.
        - rtrn : (Optional) If True, returns factorized matrices.

        Returns:
        - Factors (if rtrn=True).
        """
        print("[!!!] in NMF_ROW_BACHED")
        self.eps = np.finfo(self.A_ij.dtype).eps
        self.params.eps = self.eps
        self.init_factors()
        xht, W, X, WTW, WTX                       = 0.0, 0.0, 0.0, 0.0, 0.0
        NMF_avg, H_avg, W_avg, red0_avg, red1_avg = 0.0, 0.0, 0.0, 0.0, 0.0
        GLOBAL_NORMX = 0.0
        self.events['nmf_start'].record()
        for i in range(self.params.itr):
            dt_wt,dt_wtw,dt_wtx                                     = 0, 0, 0
            dt_HHT, dt_WHHT, dt_xht, dt_wxht, dt_reduce0,dt_reduce1 = 0, 0, 0, 0,0, 0
            if self.A_IS_BATCHED:
                if self.params.W_update:
                    #////////////////////////////  (W update)> ////////////////////////////////////////////////////////
                    self.events['start'].record()
                    HT  = self.H_d.T
                    HHT = _matmul(self.H_d, HT)                                # [k, n]@[n,k] -> HHT [k,k]
                    WTX, WTWH = [], []
                    for p in range(self.nBatch):
                        #print("[!!!!!!][FroNMF_1D_col][5] len(self.H_d) = {}".format(len(self.H_d)))
                        strm_idx  = p % self.MAXSTREAMS
                        strm_name = self.stream[strm_idx]
                        stream    = self.STREAMS[strm_name]
                        with stream:
                            Q_idx = strm_idx % self.batchQeueSize
                            stream.wait_event(self.FREE_STREAM[strm_name])
                            self.W_d[Q_idx].set(self.W_h[p]) # CopyH2D_Async W_h -> >W_d [bs, k]
                            WT  = self.W_d[Q_idx].T         # [k, bs]
                            #print("[+] shape(W_d) = {} | shape(self.W_h) = {}".format(self.W_d[Q_idx].shape, self.W_h[p].shape))
                            #print("[+] shape(H_d) = {} | shape(self.H_h) = {}".format(self.H_d.shape, HT.shape))
                            if self.A_IS_SPARSE:
                                self.X_d[Q_idx].set(self.X_per[p])
                                #print("[+] shape(X_d) = {} | shape(self.X_h) = {}".format(self.X_d[Q_idx].shape, self.X_per[p].shape))
                                self.buff_idx[Q_idx].set(self.X_idx[p])
                                self.buff_ptr[Q_idx].set(self.X_ptr[p])
                                X_d = cupyx.scipy.sparse.csr_matrix((self.X_d[Q_idx][:self.sparse_vect_size['dat'][p]],
                                    self.buff_idx[Q_idx][:self.sparse_vect_size['idx'][p]],
                                    self.buff_ptr[Q_idx][:self.sparse_vect_size['ptr'][p]]), shape=(self.grid_loc_m, self.batch_size))
                                XHT = self.MM(X_d, HT, transpA=False)        # [bs, n] @ [n, k] -> XHT [bs, k]
                            else:
                                self.X_d[Q_idx].set(self.X_per[p]) # CopyH2D_Async
                                #print("[+] shape(X_d) = {} | shape(self.X_h) = {}".format(self.X_d[Q_idx].shape, self.X_per[p].shape))
                                XHT  = self.MM(self.X_d[Q_idx], HT)
                            self.W_d[Q_idx]   = _divide( _multiply(self.W_d[Q_idx], XHT), _matmul(self.W_d[Q_idx],HHT)+self.params.eps) # [bs,k] <- ([bs,k]*[bs,k]) / ([bs,k]@[k, k] + epsi)
                            self.W_d[Q_idx].get(out=self.W_h[p])                   # CopyD2H_Async W_d -> W_h
                            #del  HT
                            #self.MEMPOOL.free_all_blocks()
                            #////////////////////////////  (H update)> ////////////////////////////////////////////////////////
                            if self.A_IS_SPARSE:
                                wtx = _matmul( WT, X_d)                           # [k, bs] @ [bs, n] -> [k, n]
                            else:
                                wtx = _matmul( WT, self.X_d[Q_idx])                           # [k, bs] @ [bs, n] -> [k, n]
                            #wtx = _multiply(H, wtx)
                            WTX.append(wtx)
                            wtw  = _matmul(WT, self.W_d[Q_idx])               # [k, bs] @ [bs, k] -> [k, k]
                            #print("[+] shape(wtw) = {} | shape(self.H_d) = {}".format(wtw.shape, self.H_d.shape))
                            wtwh = _matmul(wtw, self.H_d)                            # [k, k] @ [k, n] -> [k, n]
                            WTWH.append(wtwh)
                            self.FREE_STREAM[strm_name] = stream.record()
                    # Block the `REDUCE_STREAM` until all events occur. This does not block host.
                    # This is not required when reduction is performed in the default (Stream.null)
                    # stream unless streams are created with `non_blocking=True` flag.
                    for p in range(self.batchQeueSize):
                        strm_idx  = p % self.MAXSTREAMS
                        strm_name = self.stream[strm_idx]
                        self.REDUCE_STREAM.wait_event(self.FREE_STREAM[strm_name])
                    # Local reduces
                    WTWH, WTX = sum(WTWH), sum(WTX)
                    #device.synchronize() # Wait for Local reduce to finish
                    # Global reduce
                    self.nccl_comm.Allreduce(WTWH, WTWH, op=nccl.NCCL_SUM, stream=None)
                    self.sub_comm.barrier()
                    self.nccl_comm.Allreduce(WTX, WTX  , op=nccl.NCCL_SUM, stream=None)
                    self.sub_comm.barrier()
                    self.H_d = _divide( _multiply(self.H_d, WTX), WTWH+self.params.eps )
            else: # GPU MATRIX
                for p in range(self.grid_Nc):
                    j0, j1     = p*self.grid_J, (p+1)*self.grid_J
                    #print(f"[?] shape(self.X_per[p]) = {self.X_per[p].shape}")
                    #WTX        = _matmul(WT, self.X_per[:,j0:j1])               # [k, I] @ [I, J] -> [k, J]
                    if self.A_IS_SPARSE:
                        WTX = self.MM(self.X_per[p], self.W_d, transpA=True).T  # [I, J].T @ [I, k] -> [J, k].T -> [k, J]
                    else:
                        WTX = self.MM(WT, self.X_per[p])                        # [k, I] @ [I, J] -> [k, J]
                    WTWH       = _matmul(WTW, self.H_d[:,j0:j1])                # [k, k] @ [k, J] -> [k, J]
                    WTWH      += self.eps                                       # [k, J]
                    self.events['reduce_s'].record()
                    if self.grid_Nr > 1: 
                        self.nccl_comm.Allreduce(WTX, WTX, op=nccl.NCCL_SUM, stream=None)
                    self.events['reduce_e'].record()
                    self.events['reduce_e'].synchronize()
                    dt_reduce1 += cp.cuda.get_elapsed_time(self.events['reduce_s'], self.events['reduce_e'])
                    WTX                =  _multiply(WTX, self.H_d[:,j0:j1]) # [k, J] * [k, J] -> [k, J]
                    #WTX               =  _multiply(WTX, h)                  # [k, J] * [k, J] -> [k, J]
                    self.H_d[:, j0:j1] = WTX/WTWH # _divide(WTX, WTWH)      # [k, J] / [k, J] -> [k, J] 
                    #del WTWH, WTX
                    #self.MEMPOOL.free_all_blocks()

                self.events['end'].record()
                self.events['end'].synchronize()
                del WTWH, WTX

            if False: #i % 10 == 0:
                if self.A_IS_BATCHED:
                    self.H_h = np.maximum(self.H_h, self.eps)
                else:    
                    self.H_d = cp.maximum(self.H_d, self.eps)
                self.W_d = cp.maximum(self.W_d, self.eps)
            if i == self.params.itr - 1:
                self.events['nmf_end'].record()
                self.events['nmf_end'].synchronize()
                self.dt['NMF'] = cp.cuda.get_elapsed_time(self.events['nmf_start'], self.events['nmf_end'])
                #self.W_d, self.H_d = self.normalize_features(self.W_d, self.H_d)
                #self.normalize_features()
                if self.A_IS_SPARSE:
                    NORM =  cp.trace(_matmul(WTW, HHT))
                    NORM -= 2.0*cp.trace(_matmul(WT, XHT))
                    if (self.grid_Nr > 1):
                            nccl_comm.Allreduce(NORM, NORM, op=nccl.NCCL_SUM,stream=None)
                    GLOBAL_NORMX = GLOBAL_NORMX + NORM
                    self.recon_err = GLOBAL_NORMX.get()
                else:
                    self.relative_err()


    def FroNMF_1D_row(self,factors=None, save_factors=False, rtrn=False): # [W.I.P][B.U.G]
        r"""
        Perform 1D Matrix Factorization using the Frobenius Norm Minimization (FroNMF).

        This function updates W and H matrices for matrix factorization, where
        the original matrix A is approximated as a product of W and H matrices.

        Parameters:
        - factors (optional): External factors for initialization (if any). Not currently utilized.
        - save_factors (bool, optional): Flag to decide if intermediate factors should be saved.
        - rtrn (bool, optional): Flag to decide if factors should be returned. Not currently utilized."""

        ##Attributes Updated:
        #- eps: Machine epsilon for the datatype of A_ij to avoid division by zero.
        #- params.eps: Machine epsilon parameter.
        #- Various time tracking events and intermediate matrices for computation.

        #Notes:
        #- This function uses CUDA operations and relies heavily on streams and events
        #  to synchronize and parallelize operations for better performance.
        #- The function handles both batched and sparse data but has some
        #  placeholders which suggest further extensions or checks are planned (e.g., for KL updates).
        #- The 'W_update' parameter from params decides if W should be updated in the current iteration.

        #[Work in Progress - Some features are still being developed.]
        #[Bug Warning - There might be unresolved bugs in the code.]

        self.eps = np.finfo(self.A_ij.dtype).eps
        self.params.eps = self.eps
        self.init_factors()
        xht, W, X, WTW, WTX                       = 0.0, 0.0, 0.0, 0.0, 0.0
        NMF_avg, H_avg, W_avg, red0_avg, red1_avg = 0.0, 0.0, 0.0, 0.0, 0.0
        #self.prune = var_init(self.params, 'prune', default=True)
        #if self.prune: self.prune_all()
        self.events['nmf_start'].record()
        j0, j1 = int(self.gID*self.grid_J), int( (self.gID+1)*self.grid_J )
        for i in range(self.params.itr):
            dt_wt,dt_wtw,dt_wtx                                     = 0, 0, 0
            dt_HHT, dt_WHHT, dt_xht, dt_wxht, dt_reduce0,dt_reduce1 = 0, 0, 0, 0,0, 0
            #////////////////////////////  (W update)> ////////////////////////////////////////////////////////
            if self.params.W_update:
                self.events['start'].record()
                HT  = self.H_d.T                                             # [J, k]
                HHT = _matmul(self.H_d, HT)     # Local                      # [k, I] @ [I, k] -> [k, k]
                #self.comm.barrier()
                self.sub_comm.barrier()
                self.events['reduce_s'].record()
                self.nccl_comm.Allreduce(HHT, HHT, op=nccl.NCCL_SUM, stream=None)
                self.events['reduce_e'].record()
                self.events['reduce_e'].synchronize()
                dt_reduce0 = cp.cuda.get_elapsed_time(self.events['reduce_s'], self.events['reduce_e'])
                self.dt['allRed_HHT'] = dt_reduce0*1.0
                if self.A_IS_BATCHED:
                  for p in range(self.grid_Nr):
                    i0, i1     = p*self.grid_I, (p+1)*self.grid_I
                    #print(f"[+]!!!!!!!!!! p = {p}, i0-i1 = {i0}-{i1}")
                    W          = _asarray(self.W_h[i0:i1, :],dtype=self.A_ij.dtype)  # [I, k], In stream 1, create event w_cp
                    X          = _asarray(self.X_per[i0:i1,:],dtype=self.A_ij.dtype) # [I, J], In stream 2, create event x_cp 
                    WHHT       = _matmul(W, HHT)                   # [I, k] @ [k, k] -> [I, k], wait w_cp == True, in stream0 
                    WHHT       += self.eps                         # [I, k]
                    XHT        = _matmul(X, HT)                    # [I, J] @ [J, k] -> [I, k], wait x_cp == True, in stream0
                    XHT        = W*XHT                             # [I, k] * [I, k] -> [I, k]
                    self.events['reduce_s'].record()
                    if self.grid_Nc > 1:
                        self.nccl_comm.Allreduce(XHT, XHT, op=nccl.NCCL_SUM, stream=None)
                    self.events['reduce_e'].record()
                    self.events['reduce_e'].synchronize()
                    dt_reduce1 += cp.cuda.get_elapsed_time(self.events['reduce_s'], self.events['reduce_e'])
                    self.dt['allRed_XHT'] = dt_reduce1*1.0
                    W           =  XHT/WHHT                        # _divide(XHT, WHHT)      # [I, k] / [I, k] -> [I, k]
                    self.W_h[i0:i1, :] =  _asnumpy(W) # Local host W Update. Note W is the same everywhere
                    del XHT, WHHT
                else:
                  for p in range(self.grid_Nr):
                    i0, i1     = p*self.grid_I, (p+1)*self.grid_I
                    WHHT       = _matmul(self.W_d[i0:i1, :], HHT)           # [I, k] @ [k, k] -> [I, k] 
                    WHHT       += self.eps                                  # [I, k]
                    if self.A_IS_SPARSE:
                        XHT  = self.MM(self.X_per[p], HT, transpA=False)        # [I, J] @ [J, k] -> [I, k]
                    else:
                        #XHT  = _matmul(self.X_per[i0:i1, :], HT)            # [I, J] @ [J, k] -> [I, k]
                        XHT  = self.MM(self.X_per[p], HT)            # [I, J] @ [J, k] -> [I, k]
                    #print(f"[+] p = {p}, [i0:i1] = [{i0}:{i1}]: SHAPE(W_d) = {self.W_d[i0:i1, :].shape}, SHAPE(XHT)= {XHT.shape}")
                    XHT        = self.W_d[i0:i1, :]*XHT                     # [I, k] * [I, k] -> [I, k]
                    self.events['reduce_s'].record()
                    if self.grid_Nc > 1:
                        self.nccl_comm.Allreduce(XHT, XHT, op=nccl.NCCL_SUM, stream=None)
                    self.events['reduce_e'].record()
                    self.events['reduce_e'].synchronize()
                    dt_reduce1 += cp.cuda.get_elapsed_time(self.events['reduce_s'], self.events['reduce_e'])
                    self.dt['allRed_XHT'] = dt_reduce1*1.0
                    self.W_d[i0:i1, :] = XHT/WHHT # _divide(XHT, WHHT)      # [I, k] / [I, k] -> [I, k] 
                    del XHT, WHHT
                self.events['end'].record()
                self.events['end'].synchronize()
                del HT
                dt_w = cp.cuda.get_elapsed_time(self.events['start'],self.events['end'])  # [ms]
                self.dt['W_up'] = dt_w*1.0
            #////////////////////////////  (H update)> ////////////////////////////////////////////////////////
            self.events['start'].record()
            WTW = _zeros((self.k, self.k), dtype=self.A_ij.dtype)      # [k, k]
            WTX = _zeros((self.k, self.grid_J), dtype=self.A_ij.dtype) # [k, J]
            if self.A_IS_BATCHED:
                for p in range(self.grid_Nr):
                    i0, i1 = p*self.grid_I, (p+1)*self.grid_I
                    W  = _asarray(self.W_h[i0:i1, :],dtype=self.A_ij.dtype)  # [I, k], In stream 1, create event w_cp
                    X  = _asarray(self.X_per[i0:i1,:],dtype=self.A_ij.dtype) # [I, J], In stream 2, create event x_cp
                    WT = W.T                                           # [k, I], wait w_cp == True, in stream0
                    WTW += _matmul(WT, W)                              # [k, I] @ [I, k] -> [k, k], in stream 0
                    WTX += _matmul(WT, X)                              # [k, I] @ [I, J] -> [k, J], wait x_cp == True, in stream0
                    del  WT , W, X
            else:
                for p in range(self.grid_Nr):
                    i0, i1 = p*self.grid_I, (p+1)*self.grid_I
                    #W  = self.W_d[i0:i1, :]                           # [I, k]
                    WT = self.W_d[i0:i1, :].T                          # [k, I]
                    WTW += _matmul(WT, self.W_d[i0:i1, :])             # [k, I] @ [I, k] -> [k, k]
                    if self.A_IS_SPARSE:
                        WTX += self.MM(self.X_per[p], self.W_d[i0:i1,:], transpA=True).T # [I, J].T @ [I, k] -> [J, k].T -> [k, J]
                    else:
                        #WTX += self.MM(WT, self.X_per[i0:i1,:])       # [k, I] @ [I, J] -> [k, J]
                        WTX += self.MM(WT, self.X_per[p])
                    del  WT #,W
            WTWH    = _matmul(WTW,self.H_d)                            # [k, k] @ [k, J] -> [k, J]
            del WTW
            WTWH    += self.eps
            WTX     = _multiply(WTX,self.H_d)                          # [k, J] * [k, J] -> [k, J]
            #H       = _divide(WTX, WTW)                               # [k, J] / [k, J] -> [k, J]
            self.H_d = _divide(WTX, WTWH)                              # [k, J] / [k, J] -> [k, J]
            self.events['end'].record()
            self.events['end'].synchronize()
            del WTX #, WTW
            dt_h = cp.cuda.get_elapsed_time(self.events['start'], self.events['end'])  # [ms]
            self.dt['H_up'] = dt_h*1.0
            if self.A_IS_BATCHED:
                noSupportFor("KL update for LARGE A_ij")
            else:
                WTXWH          = _zeros((self.k, self.grid_J), dtype=self.A_ij.dtype) # WTXWH [k, J]
                for p in range(self.grid_Nr):
                    i0, i1     = p*self.grid_I, (p+1)*self.grid_I
                    XWH        = _matmul(self.W_d[i0:i1, :], self.H_d)  # XWH   = W@H       : [I, k] @ [k, J] -> [I, J]
                    if self.A_IS_SPARSE:
                        noSupportFor(" SPARSE A_ij")
                    else:
                        XWH    = self.X_per[p]/(XWH + self.eps)         # XWH   = X/(XWH+e)  : [I, J] / [I, J] + eps -> [I, J]
                        WTXWH  += _matmul(self.W_d[i0:i1, :].T, XWH)    # WTXWH = W.T@XWH@HT : [k, I] @ [I, J] -> [k, J]
                self.H_d       = self.H_d*WTXWH/X1[:, None]
                del WTXWH,XWH

        self.relative_err()

