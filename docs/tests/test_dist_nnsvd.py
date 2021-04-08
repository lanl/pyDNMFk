import sys

import os
os.environ["OMP_NUM_THREADS"] = "1"
import pyDNMFk.config as config

config.init(0)
from pyDNMFk.dist_svd import *
from pyDNMFk.dist_comm import *
import pytest


@pytest.mark.mpi
def test_nnsvd_init():
    np.random.seed(0)
    from mpi4py import MPI
    main_comm = MPI.COMM_WORLD
    if main_comm.rank == 0:
        print("NNsvd for a tall matrix")
    m, k, n = 24, 2, 16
    W = np.random.rand(m, k)
    H = np.random.rand(k, n)
    A = W @ H
    for grid in ([[2, 1]]):
        p_r, p_c = grid[0], grid[1]
        comms = MPI_comm(main_comm, p_r, p_c)
        comm1 = comms.comm
        rank = comms.rank
        size = comms.size
        args = parse()
        args.size, args.rank, args.comm1, args.comm, args.p_r, args.p_c = size, rank, comm1, comms, p_r, p_c
        args.m, args.n, args.k = m, n, 2
        args.eps = np.finfo(A.dtype).eps
        dtr_blk_shp = determine_block_params(rank, (p_r, p_c), A.shape)
        blk_indices = dtr_blk_shp.determine_block_index_range_asymm()
        A_ij = A[blk_indices[0][0]:blk_indices[1][0] + 1, blk_indices[0][1]:blk_indices[1][1] + 1]
        dsvd = DistSVD(args, A_ij)
        (W_i, H_j), err = dsvd.nnsvd(flag=1, verbose=1)
        factors = np.load('nnsvd_factors_24x16.npy',
                          allow_pickle=True).item()  # Load nnsvd results computed via  sklearn
        W, H = factors['W'], factors['H']
        assert err['recon_err_svd'] < 1e-15
        assert err['recon_err_nnsvd'] < .11
        assert np.allclose(np.vstack((comm1.allgather(W_i))), W, rtol=1e-3, atol=1e-3)

    if main_comm.rank == 0:
        print("NNsvd for a short matrix")

    m, k, n = 16, 2, 24
    W = np.random.rand(m, k)
    H = np.random.rand(k, n)
    A = W @ H
    for grid in ([[1, 2]]):
        p_r, p_c = grid[0], grid[1]
        comms = MPI_comm(main_comm, p_r, p_c)
        comm1 = comms.comm
        rank = comms.rank
        size = comms.size
        dtr_blk_shp = determine_block_params(rank, (p_r, p_c), A.shape)
        blk_indices = dtr_blk_shp.determine_block_index_range_asymm()
        A_ij = A[blk_indices[0][0]:blk_indices[1][0] + 1, blk_indices[0][1]:blk_indices[1][1] + 1]
        args = parse()
        args.size, args.rank, args.comm1, args.comm, args.p_r, args.p_c = size, rank, comm1, comms, p_r, p_c
        args.m, args.n, args.k = m, n, 2
        args.eps = np.finfo(A.dtype).eps
        dsvd = DistSVD(args, A_ij)
        (W_i, H_j), err = dsvd.nnsvd(flag=1, verbose=1)
        factors = np.load('nnsvd_factors_16x24.npy',
                          allow_pickle=True).item()  # Load nnsvd results computed via  sklearn
        W, H = factors['W'], factors['H']
        assert err['recon_err_svd'] < 1e-15
        assert err['recon_err_nnsvd'] < .11
        assert np.allclose(W_i, W, rtol=1e-3, atol=1e-3)


test_nnsvd_init()
