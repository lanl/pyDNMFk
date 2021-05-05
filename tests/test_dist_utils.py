import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"
from pyDNMFk.utils import *
from mpi4py import MPI
from scipy.io import loadmat
import pytest
from pyDNMFk.dist_comm import *


@pytest.mark.mpi
def test_dist_prune_unprune_1d():
    comm = MPI.COMM_WORLD


    for grid in ([1,2],[2,1]):
        A = loadmat('../data/wtsi.mat')['X']
        if comm.rank == 0:
            print('Data shape before zero row/column addition is ', A.shape)
        z_c = np.zeros((A.shape[0], 1))
        # Lets add 3 zero rows and cols
        A = np.hstack((z_c, A[:, :A.shape[1] // 2], z_c, A[:, A.shape[1] // 2:], z_c))
        z_r = np.zeros((1, A.shape[1]))
        A = np.vstack((z_r, A[:A.shape[0] // 2, :], z_r, A[A.shape[0] // 2:, :], z_r))
        m, n = A.shape
        if comm.rank == 0:
            print('Data shape after zero row/column addition is ', A.shape)
        if comm.rank==0: print('Working on grid=',grid)
        args = parse()
        comms = MPI_comm(comm, grid[0], grid[1])
        size = comms.size
        args.topo = '1d'
        args.size, args.rank, args.comm, args.p_r, args.p_c = size, comms.rank, comms, grid[0], grid[1]
        args.row_comm, args.col_comm, args.comm1 = comms.cart_1d_row(), comms.cart_1d_column(), comms.comm
        if np.product(grid)>2:
           args.topo = '2d'
        args.k = 4
        dtr_blk_shp = determine_block_params(comms.rank, (grid[0], grid[1]), A.shape)
        blk_indices = dtr_blk_shp.determine_block_index_range_asymm()
        A_ij = A[blk_indices[0][0]:blk_indices[1][0] + 1, blk_indices[0][1]:blk_indices[1][1] + 1]
        data_op = data_operations(A_ij, args)
        idxs = data_op.zero_idx_prune()
        #print(comms.rank,[np.where(i==False) for i in idxs])
        W_i_ = np.random.rand(data_op.params.m_loc,args.k)
        H_j_ = np.random.rand(args.k,data_op.params.n_loc)
        print('Before pruning shape of A_ij,W_i,H_j for rank=', comms.rank, 'is ', A_ij.shape, W_i_.shape, H_j_.shape)
        W_orig_shape,H_orig_shape = W_i_.shape,H_j_.shape
        A_ij,W_i,H_j = data_op.prune_all(W_i_,H_j_)
        print('After pruning shape of A_ij,W_i,H_j for rank=',comms.rank,'is ',A_ij.shape,W_i.shape,H_j.shape)
        W_i, H_j = data_op.unprune_factors(W_i, H_j)
        print('After unpruning shape of W_i,H_j for rank=',comms.rank,'is ',W_i.shape,H_j.shape)
        W_unprune_shape, H_unprune_shape = W_i.shape, H_j.shape
        assert W_orig_shape == W_unprune_shape
        assert H_orig_shape == H_unprune_shape



def main():
    test_dist_prune_unprune_1d()


if __name__ == '__main__':
    main()

