import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
from . import config
from .utils import *
from .pyDNMFk import *
from .pyDNMF import *
from .dist_comm import *
import pandas as pd


class pyDNMFk_Runner:
    def __init__(self, init="rand", itr=5000, norm="kl",
                 method="mu", verbose=False, checkpoint=False, 
                 timing_stats=False, prune=False, precision="float32",
                 perturbations=20, noise_var=0.015,
                 sill_thr=0.6, sampling="uniform", process="pyDNMF"):
        """
        

        Parameters
        ----------
        init : string, optional
            NMF initializations: rand/nnsvd. The default is "rand".
        itr : int, optional
            NMF iterations. The default is 5000.
        norm : string, optional
            Reconstruction Norm for NMF to optimize:KL/FRO. The default is "kl".
        method : string, optional
            NMF update method:MU/BCD/HALS. The default is "mu".
        verbose : bool, optional
            Verbosity flag. The default is False.
        checkpoint : bool, optional
            Enable checkpoint to track the pyNMFk state. The default is False.
        timing_stats : bool, optional
            Switch to turn on/off benchmarking. The default is False.
        prune : bool, optional
            Prune zero row/column. The default is False.
        precision : string, optional
            Precision of the data(float32/float64/float16). The default is "float32".
        perturbations : int, optional
            perturbation for NMFk. The default is 20.
        noise_var : float, optional
            Noise variance for NMFk. The default is 0.015.
        sill_thr : float, optional
            SIll Threshold for K estimation. The default is 0.6.
        sampling : string, optional
            Sampling noise for NMFk i.e uniform/poisson. The default is "uniform".
        process : string, optional
            pyDNMF/pyDNMFk. The default is "uniform".
        """
        self.init = init
        self.itr = itr
        self.norm = norm
        self.method = method
        self.verbose = verbose
        self.checkpoint = checkpoint
        self.timing_stats = timing_stats
        self.prune = prune
        self.precision = precision
        self.perturbations = perturbations
        self.noise_var = noise_var
        self.sill_thr = sill_thr
        self.sampling = sampling
        self.process = process
        
        self.fpath = None
        self.ftype = None
        self.fname = None
        self.results_path = None
        self.k_range = None
        self.step_k = None
        
        if self.process not in ["pyDNMFk", "pyDNMF"]:
            raise ValueError("process should be either pyDNMFk or pyDNMF")
        
        
        config.init(0)
        config.flag = self.timing_stats
        self.main_comm = MPI.COMM_WORLD
        self.rank = self.main_comm.rank
        
        # legacy parameters
        self.p_r = None
        self.p_c = None
        self.start_k = None
        self.end_k = None
        
        
        
    def run(self, grid:list, fpath="data/", ftype="mat", fname="A_", results_path="results/",
            k_range=[1,10], step_k=1, k=4):
        """
        Begin factorization.

        Parameters
        ----------
        grid : int
            [Now of row processors, Now of column processors].
        fpath : string, optional
            data path to read(eg: tmp/). The default is "data/".
        ftype : string, optional
            data type : mat/folder/h5. The default is "mat".
        fname : string, optional
            File name. The default is "A_".
        results_path : string, optional
            Path for saving results. The default is "results/".
        k_range : TYPE, optional
            Start and end index of K for NMFk. The default is [1,10].
        step_k : int, optional
            step for K search. The default is 1.
        k : int, optional
            feature count. The default is 4.

        Returns
        -------
        results.

        """
        if len(grid) != 2 or len(k_range) != 2: 
            raise ValueError("grid and k_range needs to be a list sized 2")
        
        self.p_r = grid[0]
        self.p_c = grid[1]    
        self.start_k = k_range[0]
        self.end_k = k_range[1]
        self.fpath = fpath
        self.ftype = ftype
        self.fname = fname
        self.results_path = results_path
        self.k_range = k_range
        self.step_k = step_k
        self.grid = grid
        self.k = k
        
            
        self.comm = MPI_comm(self.main_comm, self.grid[0], self.grid[1])
        self.comm1 = self.comm.comm
        self.col_comm = self.comm.cart_1d_column()
        self.row_comm = self.comm.cart_1d_row()
        
        '''Data read block'''
        if self.verbose and self.rank == 0:
            print("Reading data now")
        
        A_ij = data_read(self).read()
        
        if self.verbose and self.rank == 0:
            print("Reading data comlete")
        
        if self.main_comm.rank == 0 and self.verbose: 
            print('Starting '+self.process+'...')
        
        results = dict()
        '''NMF/NMFk block'''
        if self.process == "pyDNMFk":
            nopt = PyNMFk(A_ij, factors=None, params=self).fit()
            results["nopt"] = nopt
        elif self.process == "pyDNMF":
            W, H, err = PyNMF(A_ij, factors=None, params=self).fit()
            results["W"] = W
            results["H"] = H
            results["err"] = err
            
        if self.main_comm.rank == 0 and self.verbose: 
            print('Done '+self.process+'.')
            
        if self.main_comm.rank == 0 and self.timing_stats:
            if self.verbose:
                print(config.time)
            time_stats = pd.DataFrame([config.time])
            stats_path = self.results_path + 'Timing_stats.csv'
            time_stats.to_csv(stats_path)
            plot_timing_stats(stats_path, self.results_path)
        
        return results
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        