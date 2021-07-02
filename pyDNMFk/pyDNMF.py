# @author: Manish Bhattarai

from .data_io import *
from .dist_nmf import *
from .dist_svd import *
from .utils import *


class PyNMF():
    r"""
    Performs the distributed NMF decomposition of given matrix X into factors W and H

    Parameters
    ----------
        A_ij : ndarray
            Distributed Data
        factors : tuple (optional)
            Distributed factors W and H
        params : class
            Class which comprises following attributes
        params.init : str
            NMF initialization(rand/nnsvd)
        params.comm1 : object
            Global Communicator
        params.comm : object
            Modified communicator object
        params.k : int
            Rank for decomposition
        params.m : int
            Global dimensions m
        params.n : int
            Global dimensions n
        params.p_r : int
            Cartesian grid row count
        params.p_c  : int
            Cartesian grid column count
        params.row_comm : object
            Sub communicator along row
        params.col_comm : object
            Sub communicator along columns
        params.W_update : bool
            flag to set W update True/False
        params.norm : str
            NMF norm to be minimized
        params.method : str
            NMF optimization method
        params.eps : float
            Epsilon value
        params.verbose : bool
            Flag to enable/disable display results
        params.save_factors : bool
            Flag to enable/disable saving computed factors"""

    @comm_timing()
    def __init__(self, A_ij, factors=None, save_factors=False, params=None):
        self.A_ij = A_ij
        self.params = params
        self.m_loc, self.n_loc = self.A_ij.shape
        self.init = self.params.init if self.params.init else 'rand'
        if "grid" in vars(self.params) and self.params.grid:
            self.p_r, self.p_c, self.k = self.params.grid[0], self.params.grid[1], self.params.k
        else:
            self.p_r, self.p_c, self.k = self.params.p_r, self.params.p_c, self.params.k
        self.comm1 = self.params.comm1  # params['comm1']
        self.cart_1d_row, self.cart_1d_column, self.comm = self.params.row_comm, self.params.col_comm, self.params.comm  # params['row_comm'],params['col_comm'],params['main_comm']
        self.verbose = self.params.verbose if self.params.verbose else False
        self.rank = self.comm1.rank
        self.eps = np.finfo(A_ij.dtype).eps
        self.params.eps = self.eps
        self.norm = var_init(self.params,'norm',default='kl')
        self.method = var_init(self.params,'method',default='mu')
        self.prune = var_init(self.params, 'prune', default=True)
        self.save_factors = save_factors
        self.params.itr = var_init(self.params,'itr',default=5000)
        self.itr = self.params.itr
        self.W_start, self.W_end = 0,0
        self.H_start, self.H_end = 0, 0
        try:
            self.W_update = self.params.W_update
        except:
            self.params.W_update = True
        self.p = self.p_r * self.p_c
        if self.p_r != 1 and self.p_c != 1:
            self.topo = '2d'
        else:
            self.topo = '1d'
        self.params.topo = self.topo
        self.data_op = data_operations(self.A_ij, self.params)
        self.params = self.data_op.params
        if factors is not None:
            if self.topo == '1d':
                self.W_i = factors[0].astype(self.A_ij.dtype)
                self.H_j = factors[1].astype(self.A_ij.dtype)
            elif self.topo == '2d':
                self.W_ij = factors[0].astype(self.A_ij.dtype)
                self.H_ij = factors[1].astype(self.A_ij.dtype)
        else:
            self.init_factors()
        if self.prune:
            if self.topo=='2d': self.A_ij,self.W_ij,self.H_ij = self.data_op.prune_all(self.W_ij,self.H_ij)
            elif self.topo == '1d': self.A_ij,self.W_i,self.H_j = self.data_op.prune_all(self.W_i, self.H_j)




    @comm_timing()
    def init_factors(self):
        """Initializes NMF factors with rand/nnsvd method"""

        if self.init == 'rand':
            if self.topo=='2d':
                self.W_ij = np.random.rand(self.params.m_loc, self.k).astype(self.A_ij.dtype)
                self.H_ij = np.random.rand(self.k, self.params.n_loc).astype(self.A_ij.dtype)
            elif self.topo == '1d':
                if self.p_c == 1:
                    self.W_i = np.random.rand(self.m_loc, self.k).astype(self.A_ij.dtype)
                    if self.rank == 0:
                        self.H_j = np.random.rand(self.k, self.n_loc).astype(self.A_ij.dtype)
                    else:
                        self.H_j = None
                    self.H_j = self.comm1.bcast(self.H_j, root=0)

                elif self.p_r == 1:
                    self.H_j = np.random.rand(self.k, self.n_loc).astype(self.A_ij.dtype)
                    if self.rank == 0:
                        self.W_i = np.random.rand(self.m_loc, self.k).astype(self.A_ij.dtype)
                    else:
                        self.W_i = None
                    self.W_i = self.comm1.bcast(self.W_i, root=0)
        elif self.init == 'nnsvd':
            if self.topo == '1d':
                dsvd = DistSVD(self.params, self.A_ij)
                self.W_i, self.H_j = dsvd.nnsvd(flag=1, verbose=0)
            elif self.topo == '2d':
                raise Exception('NNSVD init only available for 1D topology, please try with 1d topo.')

    @comm_timing()
    def fit(self):
        r"""
        Calls the sub routines to perform distributed NMF decomposition with initialization for a given norm minimization and update method

        Returns
        -------
        W_i : ndarray
            Factor W of shape m/p_r * k
        H_j : ndarray
           Factor H of shape k * n/p_c
        recon_err : float
            Reconstruction error for NMF decomposition
        """
        for i in range(self.itr):
            if self.method.lower() == 'bcd': i = self.itr - 1
            if self.topo == '2d':
                self.W_ij, self.H_ij = nmf_algorithms_2D(self.A_ij, self.W_ij, self.H_ij, params=self.params).update()
                if i % 10 == 0:
                    self.H_ij = np.maximum(self.H_ij, self.eps)
                    self.W_ij = np.maximum(self.W_ij, self.eps)
                if i == self.itr - 1:
                    self.W_ij, self.H_ij = self.normalize_features(self.W_ij, self.H_ij)
                    self.relative_err()
                    if self.verbose == True:
                        if self.rank == 0: print('relative error is:', self.recon_err)
                    if self.save_factors:
                        data_write(self.params).save_factors([self.W_ij, self.H_ij])
                    self.comm.Free()
                    if self.prune: self.W_ij,self.H_ij = self.data_op.unprune_factors(self.W_ij, self.H_ij)
                    return self.W_ij, self.H_ij, self.recon_err
            elif self.topo == '1d':
                self.W_i, self.H_j = nmf_algorithms_1D(self.A_ij, self.W_i, self.H_j, params=self.params).update()
                if i % 10 == 0:
                    self.H_j = np.maximum(self.H_j, self.eps)
                    self.W_i = np.maximum(self.W_i, self.eps)
                if i == self.itr - 1:
                    self.W_i, self.H_j = self.normalize_features(self.W_i, self.H_j)
                    self.relative_err()
                    if self.verbose == True:
                        if self.rank == 0: print('\nrelative error is:', self.recon_err)
                    if self.save_factors:
                        data_write(self.params).save_factors([self.W_i, self.H_j])
                    if self.prune:
                        self.W_i, self.H_j = self.data_op.unprune_factors(self.W_i, self.H_j)
                    return self.W_i, self.H_j, self.recon_err

    @comm_timing()
    def normalize_features(self, Wall, Hall):
        """Normalizes features Wall and Hall"""
        Wall_norm = Wall.sum(axis=0, keepdims=True)
        if self.topo == '2d':
            Wall_norm = self.comm1.allreduce(Wall_norm, op=MPI.SUM)
        elif self.topo == '1d':
            if self.p_r != 1: Wall_norm = self.comm1.allreduce(Wall_norm, op=MPI.SUM)
        Wall /= Wall_norm+ self.eps
        Hall *= Wall_norm.T
        return Wall, Hall

    @comm_timing()
    def cart_2d_collect_factors(self):
        """Collects factors along each sub communicators"""
        self.H_j = self.cart_1d_row.allgather(self.H_ij)
        self.H_j = np.hstack((self.H_j))
        self.W_i = self.cart_1d_column.allgather(self.W_ij)
        self.W_i = np.vstack((self.W_i))

    @comm_timing()
    def relative_err(self):
        """Computes the relative error for NMF decomposition"""
        if self.topo == '2d': self.cart_2d_collect_factors()
        self.glob_norm_err = self.dist_norm(self.A_ij - self.W_i @ self.H_j)
        self.glob_norm_A = self.dist_norm(self.A_ij)
        self.recon_err = self.glob_norm_err / self.glob_norm_A

    @comm_timing()
    def dist_norm(self, X, proc=-1, norm='fro', axis=None):
        """Computes the distributed norm"""
        nm = np.linalg.norm(X, axis=axis, ord=norm)
        if proc != 1:
            nm = self.comm1.allreduce(nm ** 2)
        return np.sqrt(nm)

    @comm_timing()
    def column_err(self):
        """Computes the distributed column wise norm"""
        dtr_blk = determine_block_params(self.comm1, (self.p_r, self.p_c), (self.params.m, self.params.n))
        dtr_blk_idx = dtr_blk.determine_block_index_range_asymm()
        dtr_blk_shp = dtr_blk.determine_block_shape_asymm()
        col_err_num = np.zeros(self.params.n)
        col_err_deno = np.zeros(self.params.n)
        L_errDist_num = np.zeros(self.n_loc)
        L_errDist_deno = np.zeros(self.n_loc)
        Arecon = self.W_i @ self.H_j
        for q in range(self.A_ij.shape[1]):
            L_errDist_num[q] = np.sum((self.A_ij[:, q] - Arecon[:, q]) ** 2)
            L_errDist_deno[q] = np.sum(self.A_ij[:, q] ** 2)
        col_err_num[dtr_blk_idx[0][1]:dtr_blk_idx[0][1] + dtr_blk_shp[1]] = L_errDist_num
        col_err_deno[dtr_blk_idx[0][1]:dtr_blk_idx[0][1] + dtr_blk_shp[1]] = L_errDist_deno
        col_err_num = self.comm1.allreduce(col_err_num)
        col_err_deno = self.comm1.allreduce(col_err_deno)
        col_err = np.sqrt(col_err_num / col_err_deno)
        return col_err
