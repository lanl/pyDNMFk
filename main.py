import sys
import os
#os.environ["OMP_NUM_THREADS"] = "1"
import pyDNMFk.config as config;config.init(0)
import argparse
from pyDNMFk.utils import *
from pyDNMFk.pyDNMFk import *
from pyDNMFk.pyDNMF import *
from pyDNMFk.dist_comm import *
import pandas as pd


def parser_pyNMF(parser):
    parser.add_argument('--p_r', type=int, required=True, help='Now of row processors')
    parser.add_argument('--p_c', type=int, required=True, help='Now of column processors')
    parser.add_argument('--gpu', type=bool, default=False, help='Use GPU if true')
    parser.add_argument('--k', type=int, default=4, help='feature count')
    parser.add_argument('--fpath', type=str, default='data/', help='data path to read(eg: tmp/)')
    parser.add_argument('--ftype', type=str, default='mat', help='data type : mat/folder/h5')
    parser.add_argument('--fname', type=str, default='A_', help='File name')
    parser.add_argument('--init', type=str, default='rand', help='NMF initializations: rand/nnsvd')
    parser.add_argument('--itr', type=int, default=5000, help='NMF iterations, default:1000')
    parser.add_argument('--norm', type=str, default='kl', help='Reconstruction Norm for NMF to optimize:KL/FRO')
    parser.add_argument('--method', type=str, default='mu', help='NMF update method:MU/BCD/HALS')
    parser.add_argument('--verbose', type=str2bool, default=False)
    parser.add_argument('--results_path', type=str, default='results/', help='Path for saving results')
    parser.add_argument('--checkpoint',type=str2bool,default=False,help='Enable checkpoint to track the pyNMFk state')
    parser.add_argument('--timing_stats', type=str2bool, default=False, help='Switch to turn on/off benchmarking.')
    parser.add_argument('--prune', type=str2bool, default=False, help='Prune zero row/column.')
    parser.add_argument('--precision', type=str, default='float32', help='Precision of the data(float32/float64/float16).')

    return parser


def parser_pyNMFk(parser):
    parser.add_argument('--perturbations', type=int, default=20, help='perturbation for NMFk')
    parser.add_argument('--gpu', type=bool, default=False, help='Use GPU if true')
    parser.add_argument('--noise_var', type=float, default=0.015, help='Noise variance for NMFk')
    parser.add_argument('--start_k', type=int, default=1, help='Start index of K for NMFk')
    parser.add_argument('--end_k', type=int, default=10, help='End index of K for NMFk')
    parser.add_argument('--step_k', type=int, default=1, help='step for K search')
    parser.add_argument('--sill_thr', type=float, default=0.6, help='SIll Threshold for K estimation')
    parser.add_argument('--sampling', type=str, default='uniform', help='Sampling noise for NMFk i.e uniform/poisson')
    return parser


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Arguments for pyDNMF/pyDNMFk'
                    'To run the code for pyDNMF: mpirun -n 4 python main.py --p_r=2 --p_c=2 --k=4 -fpath=../data/')  # ArgumentParser(description='Arguments for pyNMF/pyNMFk')
    parser.add_argument('--process', type=str, default='pyDNMF', help='pyDNMF/pyDNMFk')
    parser = parser_pyNMF(parser)
    parser = parser_pyNMFk(parser)
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    config.flag = args.timing_stats  # Switch for timing stats
    '''Comm initialization block'''
    main_comm = MPI.COMM_WORLD
    rank = main_comm.rank
    comm = MPI_comm(main_comm, args.p_r, args.p_c)
    args.rank = rank
    args.comm1 = comm.comm
    args.comm = comm
    args.col_comm = comm.cart_1d_column()
    args.row_comm = comm.cart_1d_row()
    '''Data read block'''
    if rank == 0: print('Reading data now')
    A_ij = data_read(args).read()
    if rank == 0: print('Reading data complete')

    '''NMF/NMFk block'''
    if args.process == 'pyDNMFk':
        if main_comm.rank == 0: print('Starting PyDNMFk...')
        nopt = PyNMFk(A_ij, factors=None, params=args).fit()
        if main_comm.rank == 0: print('PyDNMFk done.')
    elif args.process == 'pyDNMF':
        if main_comm.rank == 0: print('Starting PyDNMF...')
        W, H, err = PyNMF(A_ij, factors=None, params=args).fit()
        if main_comm.rank == 0: print('PyDNMF done.')

    if main_comm.rank == 0 and args.timing_stats:
        print(config.time)
        time_stats = pd.DataFrame([config.time])
        stats_path = args.results_path + 'Timing_stats.csv'
        time_stats.to_csv(stats_path)
        plot_timing_stats(stats_path, args.results_path)
