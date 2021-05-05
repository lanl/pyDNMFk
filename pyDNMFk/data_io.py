# @author: Manish Bhattarai
import glob
import os

import h5py
import pandas as pd
from scipy.io import loadmat

from .utils import *


class data_read():
    r"""Class for reading data.

    Parameters
    ----------
        args : class
            Class which comprises following attributes
        fpath : str
            Directory path of file to be read
        pgrid : tuple
            Cartesian grid configuration
        ftype : str
            Type of data to read(mat/npy/csv/folder)
        fname : str
            Name of the file to read
        comm (object): comm object for distributed read

        """
    @comm_timing()
    def __init__(self, args):

        self.fpath = args.fpath
        self.pgrid = [args.p_r, args.p_c]
        self.ftype = args.ftype
        self.fname = args.fname
        self.comm = args.comm1
        self.rank = self.comm.rank
        self.precision = args.precision if args.precision else 'float32'
        self.data = 0
        if self.ftype == 'folder':
            self.file_path = self.fpath + self.fname + str(self.comm.rank) + '.npy'
        else:
            self.file_path = self.fpath + self.fname + '.' + self.ftype

    @comm_timing()
    def read(self):
        r"""Data read function"""
        return self.read_dat()

    @comm_timing()
    def read_file_npy(self):
        r"""Numpy data read function"""
        self.data = np.load(self.file_path)

    @comm_timing()
    def read_file_csv(self):
        r"""CSV data read function"""
        self.data = pd.read_csv(self.file_path, header=None).values

    @comm_timing()
    def read_file_mat(self):
        r"""mat file read function"""
        self.data = loadmat(self.file_path)['X']

    @comm_timing()
    def data_partition(
            self):
        r"""
        This function divides the input matrix into chunks as specified by grid configuration.

        Return n array of shape (nrows_i, ncols_i) where i is the index of each chunk.
        \Sum_i^n ( nrows_i * ncols_i )  = arr.size

        If arr is a 2D array, the returned array should look like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        dtr_blk_shp = determine_block_params(self.rank, self.pgrid, self.data.shape)
        blk_indices = dtr_blk_shp.determine_block_index_range_asymm()
        self.data = self.data[blk_indices[0][0]:blk_indices[1][0] + 1, blk_indices[0][1]:blk_indices[1][1] + 1]

    @comm_timing()
    def save_data_to_file(self, fpath):
        r"""This function saves the splitted data to numpy array indexed with chunk number"""
        fname = fpath + 'A_' + self.comm.rank + '.npy'
        np.save(fname, self.data)

    @comm_timing()
    def read_dat(self):
        r"""Function for reading the data and split into  chunks to be reach by each MPI rank"""
        if self.ftype == 'npy':
            self.read_file_npy()
            self.data_partition()
        elif self.ftype == 'csv' or self.ftype == 'txt':
            self.read_file_csv()
            self.data_partition()
        elif self.ftype == 'mat':
            self.read_file_mat()
            self.data_partition()
        if self.ftype == 'folder':
            self.read_file_npy()
        return self.data.astype(self.precision)


class split_files_save():
    r"""Rank 0 based data read, split and save"""

    @comm_timing()
    def __init__(self, data, pgrid, fpath):
        self.data = data
        self.pgrid = pgrid
        self.p_r = pgrid[0]
        self.p_c = pgrid[1]
        self.fpath = fpath

    @comm_timing()
    def split_files(self):
        r"""Compute the index range for each block and partition the data as per the chunk"""
        dtr_blk_idx = [determine_block_params(rank, self.pgrid, self.data.shape).determine_block_index_range_asymm() for
                       rank in range(np.product(self.pgrid))]
        self.split = [self.data[i[0][0]:i[1][0] + 1, i[0][1]:i[1][1] + 1] for i in dtr_blk_idx]

    @comm_timing()
    def save_data_to_file(self):
        r"""Function to save the chunks into numpy files"""
        s = 0
        self.split = self.split_files()
        for i in range(self.p_r * self.p_c):
            name = 'A_' + str(s) + '.npy'
            fname = self.fpath + name
            arr = self.split[s - 1]
            np.save(fname, self.data)
            s += 1


class data_write():
    r"""Class for writing data/results.

    Parameters
    ----------
        args (class): class which comprises following attributes
        results_path (str): Directory path of file to write
        pgrid (tuple): Cartesian grid configuration
        ftype (str): Type of data to read(mat/npy/csv/folder)
        comm (object): comm object for distributed read

        """
    @comm_timing()
    def __init__(self, args):

        self.p_r, self.p_c = args.p_r, args.p_c
        self.pgrid = [self.p_r, self.p_c]
        self.ftype = args.ftype
        self.comm = args.comm1
        self.params = args
        self.fpath = self.params.results_paths
        self.rank = self.comm.rank

    @comm_timing()
    def create_folder_dir(self, fpath):
        r"""Create directory if not present"""
        try:
            os.mkdir(fpath)
        except:
            pass

    @comm_timing()
    def save_factors(self, factors, reg=False):
        r"""Save the W and H factors for each MPI process"""
        self.create_folder_dir(self.fpath)
        if reg == True:
            W_factors_pth = self.fpath + 'W_reg_factors/'
            H_factors_pth = self.fpath + 'H_reg_factors/'
        else:
            W_factors_pth = self.fpath + 'W_factors/'
            H_factors_pth = self.fpath + 'H_factors/'
        self.create_folder_dir(W_factors_pth)
        self.create_folder_dir(H_factors_pth)
        if self.p_r == 1 and self.p_c != 1:
            if self.rank == 0:
                np.save(W_factors_pth + 'W.npy', factors[0])
            np.save(H_factors_pth + 'H_' + str(self.rank) + '.npy', factors[1])
        elif self.p_c == 1 and self.p_r != 1:  # Saving results for each K
            if self.rank == 0:
                np.save(H_factors_pth + 'H.npy', factors[1])
            np.save(W_factors_pth + 'W_' + str(self.rank) + '.npy', factors[0])
        else:
            np.save(H_factors_pth + 'H_' + str(self.rank) + '.npy', factors[1])
            np.save(W_factors_pth + 'W_' + str(self.rank) + '.npy', factors[0])

    @comm_timing()
    def save_cluster_results(self, params):
        r"""Save cluster results to a h5 file with rank 0"""
        if self.rank == 0:
            with h5py.File(self.fpath + 'results.h5', 'w') as hf:
                hf.create_dataset('clusterSilhouetteCoefficients', data=params['clusterSilhouetteCoefficients'])
                hf.create_dataset('avgSilhouetteCoefficients', data=params['avgSilhouetteCoefficients'])
                hf.create_dataset('L_err', data=params['L_err'])
                hf.create_dataset('L_errDist', data=params['L_errDist'])
                hf.create_dataset('avgErr', data=params['avgErr'])
                hf.create_dataset('ErrTol', data=params['recon_err'])


class read_factors():
    r"""Class for reading saved factors.

    Parameters
    ----------
        factors_path : str
            Directory path of factors to read from
        pgrid : tuple
            Cartesian grid configuration

        """
    @comm_timing()
    def __init__(self, factors_path, pgrid):
        self.factors_path = factors_path
        self.W_path = self.factors_path + 'W_reg_factors/*'
        self.H_path = self.factors_path + 'H_reg_factors/*'
        self.p_grid = pgrid
        self.load_factors()

    @comm_timing()
    def custom_read_npy(self, fpath):
        r"""Read numpy files"""
        data = np.load(fpath)
        return data

    @comm_timing()
    def read_factor(self, fpath):
        """Read factors as chunks and stack them"""
        files = glob.glob(fpath)
        data = []
        if len(files) == 1:
            data = self.custom_read_npy(files)
        else:
            for file in np.sort(files):
                data.append(self.custom_read_npy(file))
        return data, len(files)

    @comm_timing()
    def load_factors(self):
        r"""Load the final stacked factors for visualization"""
        W_data, ct_W = self.read_factor(self.W_path)
        H_data, ct_H = self.read_factor(self.H_path)
        if ct_W > 1: W_data = np.vstack((W_data))
        if ct_H > 1:
            if ct_W > 1:
                H_idxs = transform_H_index(self.p_grid).rankidx2blkidx()
                H_data = np.hstack(([H_data[i] for i in H_idxs]))
            else:
                H_data = np.hstack((H_data))
        return W_data, H_data
