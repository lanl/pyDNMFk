#@author: Manish Bhattarai
from scipy.stats import wilcoxon
from . import config
from .dist_clustering import *
from .pyNMF import *
from .plot_results import *

class sample():
    """
    Generates perturbed version of data based on sampling distribution.

    Args:
        data (ndarray, sparse matrix): Array of which to find a perturbation.
        noise_var (float): The perturbation amount.
        method (str) : Method for sampling (uniform/poisson)
        seed (float),optional : Set seed for random data generation
    """


    @comm_timing()
    def __init__(self, data, noise_var, method, seed=None):

        self.X = data
        self.noise_var = noise_var
        self.seed = seed
        if self.seed != None:
            np.random.seed(self.seed)
        self.method = method
        self.X_per = 0

    @comm_timing()
    def randM(self):
        """
        Multiplies each element of X by a uniform random number in (1-epsilon, 1+epsilon).
        """

        M = 2 * self.noise_var * np.random.random_sample(self.X.shape).astype(self.X.dtype) + self.noise_var
        M = M + 1
        self.X_per = np.multiply(self.X, M)

    @comm_timing()
    def poisson(self):
        """Resamples each element of a matrix from a Poisson distribution with the mean set by that element. Y_{i,j} = Poisson(X_{i,j}"""

        self.X_per = np.random.poisson(self.X).astype(self.X.dtype)

    @comm_timing()
    def fit(self):
        r"""
        Calls the sub routines to perform resampling on data

        Returns
        -------
        X_per : ndarry
           Perturbed version of data
        """

        if self.method == 'uniform':
            self.randM()
        elif self.method == 'poisson':
            self.poisson()
        return self.X_per


class PyNMFk():
    r"""
    Performs the distributed NMF decomposition with custom clustering for estimating hidden factors k

    Parameters:
        A_ij (ndarray) : Distributed Data
        factors (tuple), optional : Distributed factors W and H
        params (class): Class which comprises following attributes
        params.init (str) : NMF initialization(rand/nnsvd)
        params.comm1 (object): Global Communicator
        params.comm (object): Modified communicator object
        params.k (int) : Rank for decomposition
        params.m (int) : Global dimensions m
        params.n (int) : Global dimensions n
        params.p_r  (int): Cartesian grid row count
        params.p_c  (int): Cartesian grid column count
        params.row_comm (object) : Sub communicator along row
        params.col_comm (object) : Sub communicator along columns
        params.W_update (bool) : flag to set W update True/False
        params.norm (str): NMF norm to be minimized
        params.method(str): NMF optimization method
        params.eps (float) : Epsilon value
        params.verbose (bool) : Flag to enable/disable display results
        params.save_factors (bool) : Flag to enable/disable saving computed factors
        params.perturbations (int) : Number of Perturbations for clustering
        params.noise_var (float) : Set noise variance for perturbing the data
        params.sill_thr (float) : Set the sillhouette threshold for estimating K with p-test
        params.start_k (int) : Starting range for Feature search K
        params.end_k (int) : Ending range for Feature search K"""

    @comm_timing()
    def __init__(self, A_ij, factors=None, params=None):
        self.A_ij = A_ij
        self.local_m, self.local_n = self.A_ij.shape
        self.params = params
        self.comm1 = self.params.comm1
        self.rank = self.comm1.rank
        self.p_r, self.p_c = self.params.p_r, self.params.p_c
        self.fpath = self.params.fpath
        self.fname = self.params.fname
        self.p = self.p_r * self.p_c
        if self.p_r != 1 and self.p_c != 1:
            self.topo = '2d'
        else:
            self.topo = '1d'
        self.sampling = var_init(self.params,'sampling',default='uniform')
        self.perturbations = var_init(self.params,'perturbations',default=20)
        self.noise_var = var_init(self.params,'noise_var',default=.03)
        self.Hall = 0
        self.Wall = 0
        self.recon_err = 0
        self.AvgH = 0
        self.AvgG = 0
        self.col_err = 0
        self.clusterSilhouetteCoefficients, self.avgSilhouetteCoefficients = 0, 0
        self.L_errDist = 0
        self.avgErr = 0
        self.start_k = self.params.start_k  # ['start_k']
        self.end_k = self.params.end_k  # ['end_k']
        self.sill_thr = var_init(params,'sill_thr',default=0.9)
        self.verbose = var_init(params,'verbose',default=False)


    @comm_timing()
    def fit(self):
        r"""
        Calls the sub routines to perform distributed NMF decomposition and then custom clustering to estimate k

        Returns
        -------
        nopt : int
           Estimated value of latent features
        """
        SILL_MIN = []
        errRegres = []
        errRegresTol = []
        RECON = []
        RECON1 = []
        self.params.results_path = self.params.results_path + self.params.fname + '/'
        if self.rank == 0:
            try: os.makedirs(self.params.results_paths)
            except: pass
        for self.k in range(self.start_k, self.end_k + 1):
            self.params.k = self.k
            self.pynmfk_per_k()
            SILL_MIN.append(round(np.min(self.clusterSilhouetteCoefficients), 2))
            errRegres.append([self.col_err])
            errRegresTol.append([self.recon_err])
            RECON.append(self.L_errDist)
            RECON1.append(self.avgErr)
        if self.rank == 0:
            nopt1, pvalue1 = self.pvalueAnalysis(errRegres, SILL_MIN)
            print('Rank estimated by NMFk = ', nopt1)
            plot_results(self.start_k, self.end_k, RECON, RECON1, SILL_MIN, self.params.results_path, self.fname)
        else:
            nopt1 = None
        nopt1 = self.comm1.bcast(nopt1, root=0)
        self.comm1.barrier()
        return nopt1

    @comm_timing()
    def pynmfk_per_k(self):
        """Performs NMF decomposition and clustering for each k to estimate silhouette statistics"""
        self.params.results_paths = self.params.results_path+ str(self.k) + '/'
        if self.rank == 0:
            try: os.makedirs(self.params.results_paths)
            except: pass
        results = []
        if self.rank == 0: print('*************Computing for k=', self.k, '************')
        for i in range(self.perturbations):
            if self.rank == 0: print('Current perturbation =', i)
            data = sample(data=self.A_ij, noise_var=self.noise_var, method=self.sampling, seed=i * 1000).fit()
            self.params.W_update = True
            results.append(PyNMF(data, factors=None, params=self.params).fit())
        self.Wall = np.hstack(([results[i][0] for i in range(self.perturbations)]))
        self.Wall = self.Wall.reshape(self.Wall.shape[0], self.k, self.perturbations, order='F')
        self.Hall = np.vstack(([results[i][1] for i in range(self.perturbations)]))
        self.Hall = self.Hall.reshape(self.k, self.Hall.shape[1], self.perturbations)
        self.recon_err = [results[i][2] for i in range(self.perturbations)]
        [processAvg, processSTD, self.Hall, self.clusterSilhouetteCoefficients, self.avgSilhouetteCoefficients,
         idx] = custom_clustering(self.Wall, self.Hall, self.params).fit()
        self.AvgH = np.median(self.Hall, axis=-1)
        self.AvgW = processAvg
        self.params.W_update = False
        regressH = PyNMF(self.A_ij, factors=[self.AvgW, self.AvgH], params=self.params)
        self.AvgW, self.AvgH, self.L_errDist = regressH.fit()
        self.col_err = regressH.column_err()
        self.avgErr = np.mean(self.recon_err)
        cluster_stats = {'clusterSilhouetteCoefficients': self.clusterSilhouetteCoefficients,
                         'avgSilhouetteCoefficients': self.avgSilhouetteCoefficients, 'L_errDist': self.L_errDist, \
                         'L_err': self.col_err, 'avgErr': self.avgErr, 'recon_err': self.recon_err}
        data_writer = data_write(self.params)
        data_writer.save_factors([self.AvgW, self.AvgH], reg=True)
        data_writer.save_cluster_results(cluster_stats)

    @comm_timing()
    def pvalueAnalysis(self, errRegres, SILL_MIN):
        """
        Calculates nopt by analysing the errors distributions

        Parameters
        ----------
        errRegres : array
             array for storing the distributions of errors
        SILL_MIN : float
            Minimum of silhouette score
        """
        pvalue = np.ones(self.end_k - self.start_k + 1)
        oneDistrErr = errRegres[0][0];
        i = 1
        i_old = 0
        nopt = 1

        while i < (self.end_k - self.start_k + 1):
            i_next = i
            if SILL_MIN[i - 1] > self.sill_thr:  # 0.75:
                pvalue[i] = wilcoxon(oneDistrErr, errRegres[i][0])[1]
                if pvalue[i] < 0.05:
                    i_old = i
                    nopt = i
                    oneDistrErr = np.copy(errRegres[i][0])
                    i = i + 1
                else:
                    i = i + 1
            else:
                i = i + 1
        # print('nopt=', nopt)
        return nopt + self.start_k - 1, pvalue
