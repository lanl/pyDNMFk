import pytest

import sys

sys.path.append("..")
import pyDNMFk.config as config

config.init(0)
from pyDNMFk.utils import *
from pyDNMFk.dist_clustering import *
from pyDNMFk.dist_comm import *




@pytest.mark.mpi
def test_dist_clustering():
    import numpy as np
    np.random.seed(100)
    from mpi4py import MPI
    main_comm = MPI.COMM_WORLD

    def gauss(n, mean, std):
        return np.exp(-(np.linspace(1, n, n) - mean) ** 2 / std)

    p_r = 2
    p_c = 1
    comms = MPI_comm(main_comm, p_r, p_c)
    comm1 = comms.comm
    rank = comms.rank
    import numpy as np
    m, p, k = 16, 4, 3

    G1 = gauss(m, 3, 3)
    G2 = gauss(m, 8, 2)
    G3 = gauss(m, 14, 3)

    W = np.vstack([G1, G2, G3]).T
    W_all = np.stack([W[:, np.random.permutation(k)] + np.random.rand(m, k) * .1 for _ in range(p)], axis=-1)
    H_dist = np.random.rand(k, 5, p)
    if rank == 0:
        print(W_all.shape)
        W_dist = [k for k in W_all.reshape(2, 8, k, p)]
    else:
        W_dist = None
    args = parse()
    W_dist = comm1.scatter(W_dist, root=0)
    args.comm1 = comm1
    args.p_r, args.p_c, = p_r, p_c
    args.eps = np.finfo(W.dtype).eps
    cluster = custom_clustering(W_dist, H_dist, args)
    cluster.fit()
    tmp2 = cluster.dist_silhouettes()

    if rank == 0:
        print('distributed sil:', tmp2)
        sil = np.load('sill.npy')
        print('serial sil:', sil)
        assert np.allclose(sil, tmp2, rtol=1e-3, atol=1e-3)


test_dist_clustering()
