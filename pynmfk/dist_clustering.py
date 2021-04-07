# @Author: Manish Bhattarai, Erik Skau
from .utils import *


class custom_clustering():
    r"""
    Greedy algorithm to approximate a quadratic assignment problem to cluster vectors. Given p groups of k vectors, construct k clusters, each cluster containing a single vector from each of the p groups. This clustering approximation uses cos distances and mean centroids.

    Args:

        W_all (ndarray) : Order three tensor of shape m by k by p, where m is the ambient dimension of the vectors, k is the number of vectors in each group, and p is the number of groups of vectors.
        H_all (ndarray) : Order three tensor of shape n by k by p, where n is the ambient dimension of the vectors, k is the number of vectors in each group, and p is the number of groups of vectors.
        params (class) : Class object with communication parameters which comprises of grid information (p_r,p_c) , commincator (comm) and epsilon (eps).

    """

    @comm_timing()
    def __init__(self, Wall, Hall, params):
        self.W_all = Wall
        self.H_all = Hall
        self.p_r, self.p_c = params.p_r, params.p_c
        self.comm1 = params.comm1
        self.eps = params.eps
        self.p = self.p_r * self.p_c

    @comm_timing()
    def normalize_by_W(self):
        r'''Normalize the factors W and H'''
        Wall_norm = (self.W_all * self.W_all).sum(axis=0)
        if self.p_r != 1:
            Wall_norm = self.comm1.allreduce(Wall_norm)
        Wall_norm += self.eps
        temp = np.sqrt(Wall_norm)
        self.W_all /= temp.reshape(1, temp.shape[0], temp.shape[1])
        self.H_all *= temp.reshape(temp.shape[0], 1, temp.shape[1])

    @comm_timing()
    def mad(self, data, flag=1, axis=-1):
        r'''Compute the median/mean absolute deviation'''
        if flag == 1:  # the median absolute deviation
            return np.nanmedian(np.absolute(data - np.nanmedian(data, axis=axis, keepdims=True)), axis=axis)
        else:  # flag = 0 the mean absolute deviation
            # return np.nanmean((np.absolute(data.T - np.nanmean(data, axis = dimf))).T,axis = dimf)
            return np.nanmean(np.absolute(data - np.nanmean(data, axis=axis)), axis=axis)

    @comm_timing()
    def change_order(self, tens):
        r'''change the order of features'''
        ans = list(range(len(tens)))
        for p in tens:
            ans[p[0]] = p[1]
        return ans

    @comm_timing()
    def greedy_lsa(self, A):
        r"""Return the permutation order"""
        X = A.copy()
        pairs = []
        for i in range(X.shape[0]):
            minindex = np.argmax(X)
            ind = np.unravel_index(minindex, X.shape)
            pairs.append(ind)
            X[:, ind[1]] = -np.inf
            X[ind[0], :] = -np.inf
        return pairs

    @comm_timing()
    def dist_feature_ordering(self, centroids, W_sub):
        r'''return the features in proper order'''
        k = W_sub.shape[1]
        dist = centroids.T @ W_sub
        if self.p_r != 1:
            dist = self.comm1.allreduce(dist)
        tmp = self.greedy_lsa(dist)
        j = self.change_order(tmp)
        W_sub = W_sub[:, j]
        return W_sub, j

    @comm_timing()
    def dist_custom_clustering(self, centroids=None, vb=0):
        """
        Performs the distributed custom clustering

        Parameters
        ----------
        centroids : ndarray, optional
           The m by k initialization of the centroids of the clusters. None corresponds to using the first slice, W_all[:,:,0], as the initial centroids. Defaults to None.
        vb : bool, optional
           Verbose to display intermediate results

        Returns
        -------
        centroids : ndarray
           The m by k centroids of the clusters
        W_all :ndarray
           Clustered organization of the vectors W_all
        H_all : ndarray
           Clustered organization of the vectors H_all
        permute_order : list
           Indices of the permuted features
        """

        permute_order = []
        self.normalize_by_W()
        if centroids == None:
            centroids = self.W_all[:, :, 0].copy()
        dist = centroids.T @ self.W_all[:, :, 0]
        if self.p_r != 1:
            dist = self.comm1.allreduce(dist)
        for i in range(100):
            for p in range(self.W_all.shape[-1]):
                W_ord, j = self.dist_feature_ordering(centroids, self.W_all[:, :, p])
                permute_order.append(j)
                self.W_all[:, :, p] = W_ord
                self.H_all[:, :, p] = [self.H_all[:, :, p][k] for k in j]
            centroids = np.median(self.W_all, axis=-1)
            centroids_norm = (centroids ** 2).sum(axis=0)
            if self.p_r != 1:
                centroids_norm = self.comm1.allreduce(centroids_norm)
            centroids_norm += self.eps
            temp = np.sqrt(centroids_norm)
            centroids /= temp
        return centroids, self.W_all, self.H_all, permute_order

    @comm_timing()
    def dist_silhouettes(self):
        """
        Computes the cosine distances silhouettes of a distributed clustering of vectors.

        Returns
        -------
        sils : ndarray
            The k by p array of silhouettes where sils[i,j] is the silhouette measure for the vector W_all[:,i,j]
        """

        self.dist_custom_clustering()
        N, k, n_pert = self.W_all.shape
        W_flat = self.W_all.reshape(N, k * n_pert)
        W_all2 = (W_flat.T @ W_flat).reshape(k, n_pert, k, n_pert)
        if self.p_r != 1:
            W_all2 = self.comm1.allreduce(W_all2)
        distances = np.arccos(np.clip(W_all2, -1.0, 1.0))
        (N, K, n_perts) = self.W_all.shape
        if K == 1:
            sils = np.ones((K, n_perts))
        else:
            a = np.zeros((K, n_perts))
            b = np.zeros((K, n_perts))
            for k in range(K):
                for n in range(n_perts):
                    a[k, n] = 1 / (n_perts - 1) * np.sum(distances[k, n, k, :])
                    tmp = np.sum(distances[k, n, :, :], axis=1)
                    tmp[k] = np.inf
                    b[k, n] = 1 / n_perts * np.min(tmp)
            sils = (b - a) / np.maximum(a, b)
        return sils

    @comm_timing()
    def fit(self):
        r"""
        Calls the sub routines to perform distributed custom clustering and  compute silhouettes

        Returns
        -------
        centroids : ndarray
            The m by k centroids of the clusters
        CentStd : ndarray
            Absolute deviation of the features from the centroid
        W_all : ndarray
            Clustered organization of the vectors W_all
        H_all : ndarray
            Clustered organization of the vectors H_all
        S_avg : ndarray
            mean Silhouette score
        permute_order : list
            Indices of the permuted features
        """

        centroids, _, _, IDX_F2 = self.dist_custom_clustering()
        CentStd = self.mad(self.W_all, axis=-1)
        cluster_coefficients = self.dist_silhouettes()
        S_avg = cluster_coefficients.flatten().mean()
        result = [centroids, CentStd, self.H_all, cluster_coefficients.mean(axis=1), S_avg, IDX_F2]
        return result
