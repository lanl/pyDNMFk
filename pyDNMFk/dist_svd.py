# @Author: Gopinath Chennupati, Manish Bhattarai, Erik Skau
from datetime import datetime
from math import sqrt
from random import normalvariate

from .utils import *


class DistSVD():
    r"""
    Distributed Computation of SVD along 1D distribution of the data. Only U or V is distributed based on data size.

    Parameters
    ----------
        A : ndarray
            Distributed Data
        args : class
            Class which comprises following attributes
        args.globalm : int
            Global row dimensions of A
        args.globaln : int
            Global column dimension of A
        args.k : int(optional)
            Rank for decomposition
        args.p_r : int
            Cartesian grid row count
        args.p_c  : int
            Cartesian grid column count
        args.seed : int(optional)
            Set the random seed
        args.comm : object
            comm object for distributed read
        args.eps : float
            Epsilon value"""

    @comm_timing()
    def __init__(self, args, A):
        super(DistSVD, self).__init__()
        self.args = args
        self.globalm = self.args.m  # ['globalm']
        self.globaln = self.args.n  # ['globaln']
        # SVD inits
        self.k = self.args.k if self.args.k else min(self.globalm, self.globaln)
        self.svdSoFar = []

        # MPI inits
        self.comm = args.comm  # ['comm']
        self.rank = self.comm.rank
        self.p = self.comm.size
        self.grid_comm = self.comm.cartesian2d
        self.coords = self.comm.coord2d
        self.proc_rows = self.args.p_r  # num_row_ranks()
        self.proc_cols = self.args.p_c
        if self.globalm > self.globaln:
            assert self.proc_rows > self.proc_cols, "m>n , ensure p_r>p_c"
        elif self.globalm < self.globaln:
            assert self.proc_rows < self.proc_cols, "m<n , ensure p_r<p_c"
        self.eps = self.args.eps
        self.A = A
        self.mat_for_1D = self.A.copy()
        # seed
        try: self.seed = self.args.seed
        except: self.seed = datetime.now().timestamp()

        # precision
        # self._tensor_type = torch.FloatTensor # For some reason torch.cat doesn't accept double tensors

    @comm_timing()
    def normalize_by_W(self, Wall, Hall, comm1):
        """Normalize the factors W and H"""
        Wall_norm = Wall.sum(axis=0, keepdims=True)
        if self.proc_rows != 1:
            Wall_norm = comm1.allreduce(Wall_norm, op=MPI.SUM)
        Wall_norm += self.eps
        # temp = np.sqrt(Wall_norm)
        Wall /= Wall_norm
        Hall *= Wall_norm.T
        return Wall, Hall

    @comm_timing()
    def randomUnitVector(self, d):
        """
        Construnct a rondom unit vector
        """
        unnormalized = [normalvariate(0, 1) for _ in range(d)]
        theNorm = sqrt(sum(x * x for x in unnormalized))
        return np.asarray([x / theNorm for x in unnormalized], dtype='float64')

    @comm_timing()
    def globalGram(self, X, Y):
        """Compute the global gram betwee X and Y"""
        B = X @ Y
        B = self.grid_comm.allreduce(B)
        return B

    @comm_timing()
    def svd1D(self):
        """
        One dimensional SVD
        """
        buf = self.randomUnitVector(min(self.globalm, self.globaln))
        self.lastV = None
        if self.rank == 0: buf += np.zeros(buf.shape)
        self.grid_comm.Bcast(buf, 0)
        # self.currV = torch.from_numpy(buf).type(self._tensor_type)
        self.currV = buf
        # print("current Vector {}".format(self.currV))

        # self.At = self.mat_for_1D.t() # Transpose current A
        self.At = self.mat_for_1D.T
        # print("At rank {} A.t() \n{}".format(self.rank, self.At))
        if self.globalm >= self.globaln:
            self.B = self.globalGram(self.At, self.mat_for_1D)
        else:
            self.B = self.globalGram(self.mat_for_1D, self.A.T)
        if self.rank != 0:
            del self.B
        b = np.zeros(self.currV.shape)
        if self.rank == 0:
            iteration = 0
            while True:
                lastV = self.currV
                self.currV = self.B @ self.currV
                # self.currV = self.currV / torch.norm(self.currV)
                self.currV = self.currV / np.linalg.norm(self.currV)
                r = np.dot(self.currV, lastV)
                iteration += 1
                # if abs(r.item()) > 1. - eps:
                if abs(r.item()) > 1. - self.eps:
                    # print("Converged in {}".format(iteration))
                    # b = self.currV.numpy()
                    b += self.currV
                    # self.currV = torch.from_numpy(b)
                    self.currV = b
                    # return self.currV
                    break
        self.grid_comm.Bcast(self.currV, 0)

    @comm_timing()
    def calc_norm(self, vec):
        """Compute the norm of vector"""
        partial_sq_sum = sum(vec * vec)
        global_sq_sum = self.grid_comm.allreduce(partial_sq_sum)
        # norm = torch.sqrt(global_sq_sum)
        norm = np.sqrt(global_sq_sum)
        return norm

    @comm_timing()
    def svd(self):
        """
        Computes the SVD for a given matrix

        Returns
        -------
        singularValues : list
           List of singular values of length k
        Us :ndarray
           Factor Us of shape (m/p_r,k)
        Vs : ndarray
           Factor Vs of shape (k,n/p_c)"""

        for i in range(self.k):
            self.mat_for_1D = self.A.copy()

            for sigma, u, v in self.svdSoFar[:i]:
                # outer = torch.ger(u, v)
                outer = np.outer(u, v)
                self.mat_for_1D -= sigma * (outer)  # Need to fix this

            if self.globalm > self.globaln:
                self.svd1D()
                v = self.currV
                u_unnorm = self.A @ v
                sig = self.calc_norm(u_unnorm)  # next singular value
                u = u_unnorm / sig
            else:
                self.svd1D()
                u = self.currV
                v_unnorm = self.A.T @ u
                sig = self.calc_norm(v_unnorm)  # next singular value
                v = v_unnorm / sig
            # if self.rank==0: print(sig)
            self.svdSoFar.append([sig, u, v])
        singularValues, us, vs = [np.asarray(x) for x in zip(*self.svdSoFar)]
        # if self.rank==0: print("Rank ", self.rank, singularValues, us.shape, vs.shape)
        return singularValues, us.T, vs

    @comm_timing()
    def rel_error(self, U, S, V):
        """Computes the relative error between the reconstructed data with factors vs original data"""
        X_recon = U @ S @ V
        err_num = np.sum((self.A - X_recon) ** 2)
        norm_deno = np.sum(self.A ** 2)
        err_num = self.grid_comm.allreduce(err_num)
        norm_deno = self.grid_comm.allreduce(norm_deno)
        err = np.sqrt(err_num) / np.sqrt(norm_deno)
        return err

    @comm_timing()
    def nnsvd(self, flag=1, verbose=1):
        r"""
        Computes the distributed Non-Negative SVD(NNSVD) components from the computed SVD factors.

        Parameters
        ----------
        flag : bool, optional
           Computes nnSVD factors with different configurations
        verbose : bool, optional
           Verbose to set returned errors. If true returns SVD and NNSVD reconstruction errors.

        Returns
        -------
        W :ndarray
           Non-negative factor W  of shape (m/p_r,k)
        H : ndarray
           Non-negative factor H  of shape (k,n/p_c)
        error : dictionary (optional)
           Dictinoary of reconstruction error for svd and nnsvd
        """
        singularValues, U, V = self.svd()
        if verbose == 1:
            recon_err_svd = self.rel_error(U, np.diag(singularValues), V)
            if self.rank == 0:
                print('Reconstruction error for SVD is :', recon_err_svd)
        if flag == 0:
            S = np.diag(singularValues)
            W = U
            H = S @ V
            W[W < 0] = 0
            H[H < 0] = 0
            # return W,H

        elif flag == 1:
            S = singularValues.copy()
            U, S, V = U[:, :self.k], S[:self.k], V[:self.k, :]
            V = V.T
            UP = np.where(U > 0, U, 0)
            UN = np.where(U < 0, -U, 0)
            VP = np.where(V > 0, V, 0)
            VN = np.where(V < 0, -V, 0)

            UP_norm = np.sum(np.square(UP), 0)
            UP_norm = np.sqrt(self.grid_comm.allreduce(UP_norm))
            UN_norm = np.sum(np.square(UN), 0)
            UN_norm = np.sqrt(self.grid_comm.allreduce(UN_norm))
            VP_norm = np.sum(np.square(VP), 0)
            VP_norm = np.sqrt(VP_norm)
            VN_norm = np.sum(np.square(VN), 0)
            VN_norm = np.sqrt(VN_norm)
            if self.globalm > self.globaln:
                UP_norm, UN_norm = UP_norm / self.p, UN_norm / self.p
            mp = np.sqrt(UP_norm * VP_norm * S)
            mn = np.sqrt(UN_norm * VN_norm * S)

            W = np.where(mp > mn, mp * UP / (UP_norm + self.eps), mn * UN / (UN_norm + self.eps))
            H = np.where(mp > mn, mp * VP / (VP_norm + self.eps), mn * VN / (VN_norm + self.eps)).T

        if verbose == 1:
            # print(W.shape,H.shape)
            recon_err_nnsvd = self.rel_error(W, np.eye(self.k), H)
            if self.rank == 0:
                print('Reconstruction error for nnSVD is :', recon_err_nnsvd)
        if verbose == 1:
            return self.normalize_by_W(W, H, self.grid_comm), {'recon_err_svd': recon_err_svd,
                                                               'recon_err_nnsvd': recon_err_nnsvd}
        else:
            return self.normalize_by_W(W, H, self.grid_comm)
