# @author: Manish Bhattarai
from numpy import matlib

from .utils import *


class nmf_algorithms_2D():
    """
    Performs the distributed NMF operation along 2D cartesian grid

    Parameters
    ----------
        A_ij : ndarray
            Distributed Data
        W_ij : ndarray
            Distributed factor W
        H_ij : ndarray
            Distributed factor H
        params : class
            Class which comprises following attributes
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
        params.p_c : int
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


    """
    @comm_timing()
    def __init__(self, A_ij, W_ij, H_ij, params=None):
        self.params = params
        self.m, self.n, self.p_r, self.p_c, self.k = self.params.m, self.params.n, self.params.p_r, self.params.p_c, self.params.k
        self.comm1 = self.params.comm1  # ['comm1']
        self.cartesian1d_row, self.cartesian1d_column, self.comm = self.params.row_comm, self.params.col_comm, self.params.comm
        self.A_ij, self.W_ij, self.H_ij = A_ij, W_ij, H_ij
        self.eps = self.params.eps
        self.p = self.p_r * self.p_c
        self.W_update = self.params.W_update
        self.norm = self.params.norm
        self.method = self.params.method
        self.rank = self.comm1.rank
        self.local_W_m = self.W_ij.shape[0]
        self.local_H_n = self.H_ij.shape[1]

    def update(self):
        """Performs 1 step Update for factors W and H based on NMF method and corresponding norm minimization

        Returns
        -------
        W_ij : ndarray
           The m/p X k distributed factor W
        H_ij : ndarray
           The k X n/p distributed factor H
        """
        if self.norm.upper() == 'FRO':
            if self.method.upper() == 'MU':
                self.Fro_MU_update(self.W_update)
            elif self.method.upper() == 'HALS':
                self.FRO_HALS_update(self.W_update)
            elif self.method.upper() == 'BCD':
                self.FRO_BCD_update(self.W_update, itr=self.params.itr)
            else:
                raise Exception('Not a valid method: Choose (mu/hals/bcd)')
        elif self.norm.upper() == 'KL':
            if self.method.upper() == 'MU':
                self.KL_MU_update(self.W_update)
            else:
                raise Exception('Not a valid method: Choose (mu)')
        else:
            raise Exception('Not a valid norm: Choose (fro/kl)')
        return self.W_ij, self.H_ij

    @comm_timing()
    def global_gram(self, A):

        r""" Distributed gram computation

        Computes the global gram operation of matrix A
        .. math:: A^TA

        Parameters
        ----------
        A  :  ndarray


        Returns
        -------

        A_TA_glob  : ndarray
        """

        A_TA_loc = np.matmul(A.T, A)
        A_TA_glob = self.comm1.allreduce(A_TA_loc, op=MPI.SUM)
        self.comm1.barrier()
        return A_TA_glob

    @comm_timing()
    def global_mm(self, A, B):

        r""" Distributed matrix multiplication

        Computes the global matrix multiplication of matrix A and B
        .. math:: AB

        Parameters
        ----------
        A  :  ndarray
        B  :  ndarray

        Returns
        -------

        AB_glob  : ndarray
        """

        AB_loc = A @ B
        AB_glob = self.comm1.allreduce(AB_loc, op=MPI.SUM)
        self.comm1.barrier()
        return AB_glob

    '''Functions for Fro MU NMF update'''

    @comm_timing()
    def ATW_glob(self):

        r""" Distributed computation of W^TA

        Computes the global matrix multiplication of matrix W and A
        .. math:: W^TA

        Parameters
        ----------
        W  :  ndarray
        A  :  ndarray

        Returns
        -------

        Atw  : ndarray
        """

        W_i = self.cartesian1d_column.allgather(self.W_ij)
        self.cartesian1d_column.barrier()
        W_i = np.vstack((W_i))
        Y_ij = np.matmul(W_i.T, self.A_ij)
        ks = np.empty([self.k, self.local_H_n]).T.copy().astype(
            self.A_ij.dtype)  # self.n // (self.p_r * self.p_c)]).T.copy().astype(self.A_ij.dtype)
        self.cartesian1d_row.Reduce_scatter(Y_ij.T.copy(), ks, op=MPI.SUM)
        self.cartesian1d_row.barrier()
        Atw = ks.T
        return Atw

    @comm_timing()
    def AH_glob(self, H_ij=None):

        r""" Distributed computation of AH^T

        Computes the global matrix multiplication of matrix A and H
        .. math:: AH^T

        Parameters
        ----------
        A  :  ndarray
        H  :  ndarray

        Returns
        -------

        AH  : ndarray
        """

        if H_ij is None:
            H_ij = self.H_ij
        H_j = self.cartesian1d_row.allgather(H_ij)
        self.cartesian1d_row.barrier()
        H_j = np.hstack((H_j))
        V_ij = np.matmul(self.A_ij, H_j.T)
        ko, l = V_ij.shape
        sk = np.empty([self.local_W_m, self.k]).astype(
            self.A_ij.dtype)  # self.m // (self.p_r * self.p_c), self.k]).astype(self.A_ij.dtype)
        self.cartesian1d_column.Reduce_scatter(V_ij, sk, op=MPI.SUM)
        self.cartesian1d_column.barrier()
        AH = sk
        return AH

    def Fro_MU_update_H(self):

        r"""
        Frobenius norm based multiplicative update of H parameter
        Function computes updated H parameter for each mpi rank

        Parameters
        ----------
        self : object

        Returns
        -------
        self.H_ij : ndarray
        """

        W_TW = self.global_gram(self.W_ij)
        AtW = self.ATW_glob()
        HWtW = np.matmul(self.H_ij.T, W_TW) + self.eps
        self.H_ij *= AtW / HWtW.T

    def Fro_MU_update_W(self):

        r"""
        Frobenius norm based multiplicative update of W parameter
        Function computes updated H parameter for each mpi rank

        Parameters
        ----------
        self : object

        Returns
        -------
        self.W_ij : ndarray
        """

        HH_T = self.global_gram(self.H_ij.T)
        AH = self.AH_glob()
        WHTH = np.matmul(self.W_ij, HH_T) + self.eps
        self.W_ij *= AH / WHTH

    def Fro_MU_update(self, W_update=True):
        r"""
        Frobenius norm based multiplicative update of W and H parameter
        Function computes updated W and H parameter for each mpi rank

        Parameters
        ----------
        self : object

        Returns
        -------
        self.H_ij : ndarray
        self.W_ij : ndarray
        """
        if W_update == True:
            self.Fro_MU_update_W()
        self.Fro_MU_update_H()

    '''Functions for KL MU NMF update'''

    @comm_timing()
    def gather_W_H(self, gW=True, gH=True):
        r"""
        Gathers W and H factors across cartesian groups
        i.e H_ij -> H_j if gH=True and W_ij -> W_i and gW=True

        Parameters
        ----------
        gW : boolen
        gH : boolen

        Returns
        -------
        self.H_j : ndarray
        self.W_i : ndarray
        """

        if gH == True:
            self.H_j = self.cartesian1d_row.allgather(self.H_ij)
            self.cartesian1d_row.barrier()
            self.H_j = np.hstack((self.H_j))
        if gW == True:
            self.W_i = self.cartesian1d_column.allgather(self.W_ij)
            self.cartesian1d_column.barrier()
            self.W_i = np.vstack((self.W_i))

    @comm_timing()
    def WTU_glob(self):
        r""" Distributed computation of W^TU

        Computes the global matrix multiplication of matrix W and U for KL
        .. math:: W^TU

        Parameters
        ----------
        W  :  ndarray
        H  :  ndarray
        A  :  ndarray

        Returns
        -------

        WTU  : ndarray
        """

        U_ij = self.A_ij / (self.W_i.dot(self.H_j) + self.eps)
        WTU = self.W_i.T.dot(U_ij)
        ks = np.empty([self.k, self.local_H_n]).T.copy().astype(self.A_ij.dtype)
        self.cartesian1d_row.Reduce_scatter(WTU.T.copy(), ks, op=MPI.SUM)
        self.cartesian1d_row.barrier()
        ks = ks.T
        return ks

    @comm_timing()
    def UHT_glob(self):
        r""" Distributed computation of UH^T

        Computes the global matrix multiplication of matrix W and U for KL
        .. math:: UH^T

        Parameters
        ----------
        W  :  ndarray
        H  :  ndarray
        A  :  ndarray

        Returns
        -------

        UHT  : ndarray
        """
        U_ij = self.A_ij / (self.W_i.dot(self.H_j) + self.eps)
        UHT = U_ij.dot(self.H_j.T)
        sk = np.empty([self.local_W_m, self.k]).astype(self.A_ij.dtype)
        self.cartesian1d_column.Reduce_scatter(UHT, sk, op=MPI.SUM)
        self.cartesian1d_column.barrier()
        return sk

    @comm_timing()
    def sum_axis(self, dat, axis):
        tmp = dat.sum(axis=axis)
        tmp = self.comm1.allreduce(tmp, op=MPI.SUM)
        return tmp

    def KL_MU_update_W(self):
        r"""
        KL divergence based multiplicative update of W parameter
        Function computes updated W parameter for each mpi rank

        Parameters
        ----------
        self : object

        Returns
        -------
        self.W_ij : ndarray
            Distributed factor W of shape m/p X k
        """
        x2 = self.sum_axis(self.H_ij, axis=1)
        X2 = matlib.repmat(x2, self.local_W_m, 1)
        self.gather_W_H()
        sk = self.UHT_glob()
        self.W_ij *= sk / (X2 + self.eps)

    def KL_MU_update_H(self):
        r"""
        Frobenius norm based multiplicative update of H parameter
        Function computes updated H parameter for each mpi rank

        Parameters
        ----------
        self : object

        Returns
        -------
        self.H_ij : ndarray
            Distributed factor H of shape  k X n/p
        """
        x1 = self.sum_axis(self.W_ij, axis=0)
        X1 = matlib.repmat(x1, self.local_H_n, 1).T
        self.gather_W_H()
        ks = self.WTU_glob()
        self.H_ij *= ks / (X1 + self.eps)

    def KL_MU_update(self, W_update=True):
        r"""
        KL divergence based multiplicative update of W and H parameter
        Function computes updated W and H parameter for each mpi rank

        Parameters
        ----------
        self : object

        Returns
        -------
        self.H_ij : ndarray (k X n/p)
        self.W_ij : ndarray (m/p X k)
        """
        if W_update == True:
            self.KL_MU_update_W()
        self.KL_MU_update_H()

    '''Functions for FRO HALS NMF update'''

    def FRO_HALS_update_W(self):
        r"""
        Frobenius norm minimization based HALS update of W  parameter
        Function computes updated W parameter for each mpi rank

        Parameters
        ----------
        self : object

        Returns
        -------
        self.W_ij : ndarray (m/p X k)
        """
        self.gather_W_H(gW=False)
        HHT = self.global_gram(self.H_ij.T)
        AH = self.AH_glob()
        for kk in iter(range(0, self.k)):
            temp_vec = self.W_ij[:, kk] * HHT[kk, kk] + AH[:, kk] - self.W_ij.dot(HHT[:, kk])
            self.W_ij[:, kk] = np.maximum(temp_vec, self.eps)
            ss = norm(self.W_ij[:, kk], self.comm1, norm=2, p=self.p_r)
            if ss > 0:
                self.W_ij[:, kk] /= ss

    def FRO_HALS_update_H(self):
        r"""
        Frobenius norm minimization based HALS update of H  parameter
        Function computes updated H parameter for each mpi rank

        Parameters
        ----------
        self : object

        Returns
        -------
        self.H_ij : ndarray ( k X n/p)
        """
        self.gather_W_H(gH=False)
        WTW = self.global_gram(self.W_ij)
        AtW = self.ATW_glob()
        for kk in iter(range(0, self.k)):
            temp_vec = self.H_ij[kk, :] + AtW[kk, :] - WTW[kk, :].dot(self.H_ij)
            self.H_ij[kk, :] = np.maximum(temp_vec, self.eps)

    def FRO_HALS_update(self, W_update=True):
        r"""
        Frobenius norm minimization based HALS update of W  and H parameter
        Function computes updated W and H parameter for each mpi rank

        Parameters
        ----------
        self : object

        Returns
        -------
        self.W_ij : ndarray (m/p X k)
        self.H_ij : ndarray (k X n/p)
        """
        if W_update == True:
            self.FRO_HALS_update_W()
        self.FRO_HALS_update_H()

    '''Functions for FRO BCD NMF update'''

    @comm_timing()
    def globalSqNorm(self, comm, X):
        """ Calc global squared norm of any matrix"""
        normX = np.linalg.norm(X)
        sqnormX = normX * normX
        Xnorm = self.comm1.allreduce(sqnormX, op=MPI.SUM)
        return Xnorm

    @comm_timing()
    def initWandH(self):
        """ Initialize the parameters for BCD updates"""

        # global Frobenius norm u calculated before
        Xnorm = self.globalSqNorm(self.comm1, self.A_ij)
        # Norm of W0 per processor
        globalsqnormW = self.globalSqNorm(self.comm1, self.W_ij)
        # Norm of H0 per processor
        globalsqnormH = self.globalSqNorm(self.comm1, self.H_ij)
        # Now normalize the initial W and H
        W_old = self.W_ij / np.sqrt(globalsqnormW) * np.sqrt(np.sqrt(Xnorm))
        H_old = self.H_ij / np.sqrt(globalsqnormH) * np.sqrt(np.sqrt(Xnorm))
        Wm = W_old.copy()
        Hm = H_old.copy()
        HHT = self.global_gram(H_old.T)
        self.H_ij = H_old
        AHT = self.AH_glob()
        obj_old = 0.5 * Xnorm  # This is correct, already a squared norm
        return Wm, Hm, HHT, AHT, W_old, H_old, obj_old, Xnorm

    def FRO_BCD_update(self, W_update=True, itr=1000):
        r"""
        Frobenius norm minimization based BCD update of W  and H parameter
        Function computes updated W and H parameter for each mpi rank

        Parameters
        ----------
          self : object

        Returns
        -------
          self.W_ij : ndarray (m/p X k)

          self.H_ij : ndarray (k X n/p)

        """
        Wm, Hm, HHT, AHT, W_old, H_old, obj_old, Xnorm = self.initWandH()
        self.params.rw = 1
        nstall = 0
        t_old = 1
        HHTnorm = 1
        WTWnorm = 1
        opts = {'rel_err': [], 'obj': []}
        max_iters = itr  # Read it from the arguments or somewhere
        # relerr1 = relerr2 = []
        # Iterate for max iterations:
        for i in range(max_iters):
            """
            W update
            """
            HHTnorm_old = HHTnorm  #
            HHTnorm = np.linalg.norm(HHT)  # save and update Lipschitz bound for W
            WmHHT = Wm @ HHT  # No need of dist multiplication
            GW = WmHHT - AHT  # gradient at X=Xm
            self.W_ij = np.maximum(0, Wm - GW / HHTnorm)
            # L1 norm of Each column in W
            localWsum = np.sum(self.W_ij, 0, keepdims=True)
            globalWSum = self.comm1.allreduce(localWsum, op=MPI.SUM)
            self.comm1.barrier()
            self.W_ij /= globalWSum
            WTW = self.global_gram(self.W_ij)
            """
            H update
            """
            WTWnorm_old = WTWnorm
            WTWnorm = np.linalg.norm(WTW)  # save and update Lipschitz bound for H
            WTWHm = WTW @ Hm  # No need of dist multiplication
            WTA = self.ATW_glob()
            GH = WTWHm - WTA  # gradient at Y=Ym
            self.H_ij = np.maximum(0, Hm - GH / WTWnorm)
            HHT = self.global_gram(self.H_ij.T)
            AHT = self.AH_glob()
            self.gather_W_H()
            glob_jt = self.globalSqNorm(self.comm1, self.A_ij - self.W_i @ self.H_j)
            obj = 0.5 * glob_jt  #
            rel_err = np.sqrt(2 * obj / Xnorm)
            opts['obj'].append(obj)
            opts['rel_err'].append(rel_err)
            # --- correction and extrapolation ---
            t = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
            if obj >= obj_old:
                # restore to previous W,H, and cached quantities for nonincreasing objective
                Wm = W_old.copy()
                Hm = H_old.copy()
                HHT = self.global_gram(H_old.T)
                AHT = self.AH_glob(H_old)
            else:
                # extrapolation
                w = (t_old - 1) / t  # extrapolation weight
                ww = np.min([w, self.params.rw * np.sqrt(HHTnorm_old / HHTnorm)])  # choose smaller one for convergence
                wh = min([w, self.params.rw * np.sqrt(WTWnorm_old / WTWnorm)])
                Wm = self.W_ij + ww * (self.W_ij - W_old)
                Hm = self.H_ij + wh * (self.H_ij - H_old)  # extrapolation
                W_old = self.W_ij.copy()
                H_old = self.H_ij.copy()
                t_old = t
                obj_old = obj


class nmf_algorithms_1D():
    """
    Performs the distributed NMF operation along 1D cartesian grid

    Parameters
    ----------
        A_ij : ndarray
            Distributed Data
        W_i : ndarray
            Distributed factor W
        H_j : ndarray
            Distributed factor H
        params : class
            Class which comprises following attributes
        params.comm1 : object
            Global Communicator
        params.k : int
            Rank for decomposition
        params.m : int
            Global dimensions m
        params.n : int
            Global dimensions n
        params.p_r : int
            Cartesian grid row count
        params.p_c : int
            Cartesian grid column count
        params.W_update : bool
            flag to set W update True/False
        params.norm : str
            NMF norm to be minimized
        params.method : str
            NMF optimization method
        params.eps : float
            Epsilon value """


    def __init__(self, A_ij, W_i, H_j, params=None):
        self.m, self.n, self.p_r, self.p_c, self.k = params.m, params.n, params.p_r, params.p_c, params.k
        self.params = params
        self.comm = self.params.comm1  # ['comm1']
        # self.comm = comm
        self.norm = self.params.norm
        self.method = self.params.method
        self.comm1 = self.params.comm1
        self.A_ij, self.W_i, self.H_j = A_ij, W_i, H_j
        self.eps = self.params.eps
        self.p = self.p_r * self.p_c
        self.W_update = self.params.W_update
        self.rank = self.comm1.rank
        self.local_W_m = self.W_i.shape[0]
        self.local_H_n = self.H_j.shape[1]

    def update(self):
        """Performs 1 step Update for factors W and H based on NMF method and corresponding norm minimization

         Returns
         -------
         W_i : ndarray
            The m/p_r X k distributed factor W
         H_j : ndarray
            The k X n/p_c distributed factor H
         """
        if self.norm.upper() == 'FRO':
            if self.method.upper() == 'MU':
                self.Fro_MU_update(self.W_update)
            elif self.method.upper() == 'HALS':
                self.FRO_HALS_update(self.W_update)
            elif self.method.upper() == 'BCD':
                self.FRO_BCD_update(self.W_update, itr=self.params.itr)
            else:
                raise Exception('Not a valid method: Choose (mu/hals/bcd)')
        elif self.norm.upper() == 'KL':
            if self.method.upper() == 'MU':
                self.KL_MU_update(self.W_update)
            else:
                raise Exception('Not a valid method: Choose (mu)')
        else:
            raise Exception('Not a valid norm: Choose (fro/kl)')
        return self.W_i, self.H_j

    @comm_timing()
    def global_gram(self, A, p=1):
        r""" Distributed gram computation

        Computes the global gram operation of matrix A
        .. math:: A^TA

        Parameters
        ----------
        A  :  ndarray
        p  : Processor count

        Returns
        -------

        A_TA_glob  : ndarray
        """
        A_TA_loc = np.matmul(A.T, A)
        if p != 1:
            A_TA_glob = self.comm1.allreduce(A_TA_loc, op=MPI.SUM)
            self.comm1.barrier()
        else:
            A_TA_glob = A_TA_loc
        return A_TA_glob

    @comm_timing()
    def global_mm(self, A, B, p=-1):
        r""" Distributed matrix multiplication

        Computes the global matrix multiplication of matrix A and B
        .. math:: AB

        Parameters
        ----------
        A  :  ndarray
        B  :  ndarray
        p  : processor count

        Returns
        -------

        AB_glob  : ndarray
        """
        AB_loc = np.matmul(A, B)
        if p != 1:
            AB_glob = self.comm1.allreduce(AB_loc, op=MPI.SUM)
            self.comm1.barrier()
        else:
            AB_glob = AB_loc
        return AB_glob

    '''Functions for Fro MU NMF update'''

    @comm_timing()
    def Fro_MU_update_W(self):
        r"""
         Frobenius norm based multiplicative update of W parameter
         Function computes updated H parameter for each mpi rank

         Parameters
         ----------
         self : object

         Returns
         -------
         self.W_i : ndarray
         """

        W_TW = self.global_gram(self.W_i, p=self.p_r)
        AtW = self.global_mm(self.W_i.T, self.A_ij, p=self.p_r)
        HWtW = np.matmul(self.H_j.T, W_TW) + self.eps
        self.H_j *= AtW / HWtW.T

    @comm_timing()
    def Fro_MU_update_H(self):
        r"""
        Frobenius norm based multiplicative update of H parameter
        Function computes updated H parameter for each mpi rank

        Parameters
        ----------
        self : object

        Returns
        -------
        self.H_j : ndarray"""

        HH_T = self.global_gram(self.H_j.T, p=self.p_c)
        AH = self.global_mm(self.A_ij, self.H_j.T, p=self.p_c)
        WHTH = np.matmul(self.W_i, HH_T) + self.eps
        self.W_i *= AH / WHTH

    @comm_timing()
    def Fro_MU_update(self, W_update=True):
        r"""
        Frobenius norm based multiplicative update of W and H parameter
        Function computes updated W and H parameter for each mpi rank

        Parameters
        ----------
        self : object

        Returns
        -------
        self.H_ij : ndarray
        self.W_ij : ndarray"""

        if W_update == True:
            self.Fro_MU_update_W()
        self.Fro_MU_update_H()

    '''Functions for KL MU NMF update'''

    @comm_timing()
    def sum_along_axis(self, X, p=1, axis=0):
        r"""
        Performs sum of the matrix along given axis

        Parameters
        ----------
        X : ndarray
            Data
        p : int
            Processor count
        axis : int
            Axis along which the sum is to be performed

        Returns
        -------
        global_axis_sum : ndarray
            Vector array after summation operation along axis"""
        loc_axis_sum = X.sum(axis=axis)
        if axis == 1:
            loc_axis_sum = loc_axis_sum.T
        if p != 1:
            glob_axis_sum = self.comm1.allreduce(loc_axis_sum, op=MPI.SUM)
            self.comm1.barrier()
        else:
            glob_axis_sum = loc_axis_sum
        return glob_axis_sum

    @comm_timing()
    def glob_UX(self, axis):
        """Perform a global operation UX for W and H update with KL"""
        UX = self.A_ij / (self.W_i @ self.H_j + self.eps)
        if axis == 1:
            UX = self.global_mm(self.W_i.T, UX, p=self.p_r)
        elif axis == 0:
            UX = self.global_mm(UX, self.H_j.T, p=self.p_c)
        return UX

    def KL_MU_update_W(self):
        r"""
        KL divergence based multiplicative update of W parameter
        Function computes updated W parameter for each mpi rank

        Parameters
        ----------
        self : object

        Returns
        -------
        self.W_i : ndarray
            Distributed factor W of shape m/p_r X k
        """
        x2 = self.sum_along_axis(self.H_j, p=self.p_c, axis=1)
        X2 = matlib.repmat(x2, self.local_W_m, 1)
        sk = self.glob_UX(axis=0)
        self.W_i *= sk / (X2 + self.eps)

    def KL_MU_update_H(self):
        r"""
        KL divergence based multiplicative update of H parameter
        Function computes updated H parameter for each mpi rank

        Parameters
        ----------
        self : object

        Returns
        -------
        self.H_j : ndarray
            Distributed factor H of shape  k X n/p_c
        """
        x2 = self.sum_along_axis(self.W_i, p=self.p_r, axis=0)
        X2 = matlib.repmat(x2, self.local_H_n, 1).T
        sk = self.glob_UX(axis=1)
        self.H_j *= sk / (X2 + self.eps)

    def KL_MU_update(self, W_update=True):
        r"""
        KL divergence based multiplicative update of W and H parameter
        Function computes updated W and H parameter for each mpi rank

        Parameters
        ----------
        W_update : bool
            Flag to enable/disable W_update


        Returns
        -------
        self.H_j : ndarray (k X n/p_r)
        self.W_i : ndarray (m/p_c X k)
        """
        if W_update == True:
            self.KL_MU_update_W()
        self.KL_MU_update_H()

    '''Functions for FRO HALS NMF update'''

    def FRO_HALS_update_W(self):
        r"""
        Frobenius norm minimization based HALS update of W  parameter
        Function computes updated W parameter for each mpi rank

        Parameters
        ----------
        self : object

        Returns
        -------
        self.W_i : ndarray (m/p_r X k)
        """
        HHT = self.global_gram(self.H_j.T, p=self.p_c)
        AH = self.global_mm(self.A_ij, self.H_j.T, p=self.p_c)
        for kk in iter(range(0, self.k)):
            temp_vec = self.W_i[:, kk] * HHT[kk, kk] + AH[:, kk] - self.W_i.dot(HHT[:, kk])
            self.W_i[:, kk] = np.maximum(temp_vec, self.eps)
            ss = norm(self.W_i[:, kk], self.comm1, norm=2, p=self.p_r)
            if ss > 0:
                self.W_i[:, kk] /= ss

    def FRO_HALS_update_H(self):
        r"""
        Frobenius norm minimization based HALS update of H  parameter
        Function computes updated H parameter for each mpi rank

        Parameters
        ----------
        self : object

        Returns
        -------
        self.H_j : ndarray ( k X n/p_c)
        """
        WTW = self.global_gram(self.W_i, p=self.p_r)
        AtW = self.global_mm(self.W_i.T, self.A_ij, p=self.p_r)
        # self.H_j = self.H_j.T
        for kk in iter(range(0, self.k)):
            temp_vec = self.H_j[kk, :] + AtW[kk, :] - WTW[kk, :].dot(self.H_j)
            self.H_j[kk, :] = np.maximum(temp_vec, self.eps)
        # self.H_j = self.H_j.T

    def FRO_HALS_update(self, W_update=True):
        r"""
        Frobenius norm minimizatio based HALS update of W and H parameter
        Function computes updated W and H parameter for each mpi rank

        Parameters
        ----------
        W_update : bool
            Flag to enable/disable W_update


        Returns
        -------
        self.H_j : ndarray (k X n/p_r)
        self.W_i : ndarray (m/p_c X k)
        """
        if W_update == True:
            self.FRO_HALS_update_W()
        self.FRO_HALS_update_H()


    '''Functions for FRO BCD NMF update'''

    @comm_timing()
    def globalSqNorm(self, X, p=-1):
        """ Calc global squared norm of any matrix"""
        normX = np.linalg.norm(X)
        sqnormX = normX * normX
        if p != 1:
            Xnorm = self.comm1.allreduce(sqnormX, op=MPI.SUM)
            self.comm1.barrier()
        else:
            Xnorm = sqnormX
        return Xnorm

    @comm_timing()
    def initWandH(self):
        """ Initialize the parameters for BCD updates"""

        # global Frobenius norm u calculated before
        Xnorm = self.globalSqNorm(self.A_ij)
        # Norm of W0 per processor
        globalsqnormW = self.globalSqNorm(self.W_i, p=self.p_r)
        # Norm of H0 per processor
        globalsqnormH = self.globalSqNorm(self.H_j, p=self.p_c)
        # Now normalize the initial W and H
        W_old = self.W_i / np.sqrt(globalsqnormW) * np.sqrt(np.sqrt(Xnorm))
        H_old = self.H_j / np.sqrt(globalsqnormH) * np.sqrt(np.sqrt(Xnorm))
        Wm = W_old.copy()
        Hm = H_old.copy()
        HHT = self.global_gram(H_old.T, p=self.p_c)
        AHT = self.global_mm(self.A_ij, H_old.T, p=self.p_c)
        obj_old = 0.5 * Xnorm  # This is correct, already a squared norm
        return Wm, Hm, HHT, AHT, W_old, H_old, obj_old, Xnorm

    def FRO_BCD_update(self, W_update=True, itr=1000):
        r"""
        Frobenius norm minimization based BCD update of W  and H parameter
        Function computes updated W and H parameter for each mpi rank

        Parameters
        ----------
           W_update: bool
            flag to enable/disable W update

        Returns
        -------
           self.W_i : ndarray (m/p_r X k)
           self.H_j : ndarray (k X n/p_c)
        """
        Wm, Hm, HHT, AHT, W_old, H_old, obj_old, Xnorm = self.initWandH()
        self.params.rw = 1
        nstall = 0
        t_old = 1
        HHTnorm = 1
        WTWnorm = 1
        opts = {'rel_err': [], 'obj': []}
        max_iters = itr  # Read it from the arguments or somewhere
        # relerr1 = relerr2 = []
        # Iterate for max iterations:
        for i in range(max_iters):
            """
            W update
            """
            HHTnorm_old = HHTnorm  #
            HHTnorm = np.linalg.norm(HHT)  # save and update Lipschitz bound for W
            WmHHT = Wm @ HHT  # No need of dist multiplication
            GW = WmHHT - AHT  # gradient at X=Xm
            self.W_i = np.maximum(0, Wm - GW / HHTnorm)
            # L1 norm of Each column in W
            localWsum = np.sum(self.W_i, 0, keepdims=True)
            if self.p_r != 1:
                globalWSum = self.comm1.allreduce(localWsum, op=MPI.SUM)
                self.comm1.barrier()
            if self.p_r == 1: globalWSum = localWsum
            self.W_i /= globalWSum
            WTW = self.global_gram(self.W_i, p=self.p_r)
            """
            H update
            """
            WTWnorm_old = WTWnorm
            WTWnorm = np.linalg.norm(WTW)  # save and update Lipschitz bound for H
            WTWHm = WTW @ Hm  # No need of dist multiplication
            WTA = self.global_mm(self.W_i.T, self.A_ij, p=self.p_r)
            GH = WTWHm - WTA  # gradient at Y=Ym
            self.H_j = np.maximum(0, Hm - GH / WTWnorm)
            HHT = self.global_gram(self.H_j.T, p=self.p_c)
            AHT = self.global_mm(self.A_ij, self.H_j.T, p=self.p_c)
            glob_jt = self.globalSqNorm(self.A_ij - self.W_i @ self.H_j)
            obj = 0.5 * glob_jt  #
            rel_err = np.sqrt(2 * obj / Xnorm)
            opts['obj'].append(obj)
            opts['rel_err'].append(rel_err)
            # --- correction and extrapolation ---
            t = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
            if obj >= obj_old:
                # restore to previous W,H, and cached quantities for nonincreasing objective
                Wm = W_old.copy()
                Hm = H_old.copy()
                HHT = self.global_gram(H_old.T, p=self.p_c)
                AHT = self.global_mm(self.A_ij, H_old.T, p=self.p_c)
            else:
                # extrapolation
                w = (t_old - 1) / t  # extrapolation weight
                ww = np.min([w, self.params.rw * np.sqrt(HHTnorm_old / HHTnorm)])  # choose smaller one for convergence
                wh = min([w, self.params.rw * np.sqrt(WTWnorm_old / WTWnorm)])
                Wm = self.W_i + ww * (self.W_i - W_old)
                Hm = self.H_j + wh * (self.H_j - H_old)  # extrapolation
                W_old = self.W_i.copy()
                H_old = self.H_j.copy()
                t_old = t
                obj_old = obj
