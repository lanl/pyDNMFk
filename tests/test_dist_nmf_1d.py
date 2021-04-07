import sys
import pytest
import os
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append("..")
import pyDNMFk.config as config

config.init(0)
from pyDNMFk.pyDNMF import *
from pyDNMFk.dist_comm import *


@pytest.mark.mpi
def test_dist_nmf_1d():
    np.random.seed(100)
    comm = MPI.COMM_WORLD
    m, k, n = 24, 2, 12
    W = np.random.rand(m, k)
    H = np.random.rand(k, n)

    A = W @ H

    for grid in ([1, 2], [2, 1]):

        p_r, p_c = grid[0], grid[1]
        comms = MPI_comm(comm, p_r, p_c)
        comm1 = comms.comm
        rank = comm.rank
        size = comm.size
        args = parse()
        args.size, args.rank, args.comm1, args.comm, args.p_r, args.p_c = size, rank, comm1, comms, p_r, p_c
        args.m, args.n, args.k = m, n, k
        args.itr, args.init = 2000, 'rand'
        args.row_comm, args.col_comm, args.comm1 = comms.cart_1d_row(), comms.cart_1d_column(), comm1
        args.verbose = True
        dtr_blk_shp = determine_block_params(rank, (p_r, p_c), A.shape)
        blk_indices = dtr_blk_shp.determine_block_index_range_asymm()
        A_ij = A[blk_indices[0][0]:blk_indices[1][0] + 1, blk_indices[0][1]:blk_indices[1][1] + 1]
        for mthd in ['mu', 'bcd', 'hals']:  # Frobenius norm, KL divergence,  and BCD implementation
            for norm in ['fro', 'kl']:
                args.method, args.norm = mthd, norm
                if norm == 'kl' and mthd != 'mu':
                    continue
                W_ij, H_ij, rel_error = PyNMF(A_ij, factors=None, params=args).fit()
                if rank == 0: print('working on grid=', grid, 'with norm = ', norm, ' method= ', mthd, 'rel error=',
                                    rel_error)
                assert rel_error < 1e-3


def main():
    test_dist_nmf_1d()


if __name__ == '__main__':
    main()
