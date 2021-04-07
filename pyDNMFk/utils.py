# @author: Manish Bhattarai, Ben Nebgen, Erik Skau
import copy
from collections import Counter

from . import config
config.init(0)
import numpy
import numpy as np
from mpi4py import MPI


class determine_block_params():
    """Computes the parameters  for each chunk to be read by MPI process

    Parameters
    ----------
    comm : object
       MPI communicator object
    pgrid : tuple
       Cartesian grid configuration
    shape : tuple
        Data shape
    """
    def __init__(self, comm, pgrid, shape):
        if type(comm) == int:
            self.rank = comm
        else:
            self.rank = comm.rank
        self.pgrid = pgrid
        self.shape = shape

    def determine_block_index_range_asymm(self):
        '''Determines the start and end indices for the Data block for each rank'''
        chunk_ind = np.unravel_index(self.rank, self.pgrid)
        start_inds = [i * (n // k) + min(i, n % k) for n, k, i in zip(self.shape, self.pgrid, chunk_ind)]
        end_inds = [(i + 1) * (n // k) + min((i + 1), n % k) - 1 for n, k, i in zip(self.shape, self.pgrid, chunk_ind)]
        return start_inds, end_inds

    def determine_block_shape_asymm(self):
        '''Determines the shape for the Data block for each rank'''
        start_inds, end_inds = self.determine_block_index_range_asymm()
        return [(j - i + 1) for (i, j) in zip(start_inds, end_inds)]


class data_operations():
    """Performs various operations on the data

    Parameters
    ----------
    data : ndarray
       Data to operate on"""
    def __init__(self, data):
        self.ten = data

    def cutZero(self, thresh=1e-8):
        """Prunes zero columns from the data"""
        tenS = list(self.ten.shape)
        dim = len(tenS)
        axSum = []
        axSel = []
        axInd = []
        for curD in range(dim):
            axisList = list(range(len(self.ten.shape)))
            axisList.pop(curD)
            axSum.append(numpy.sum(self.ten, axis=tuple(axisList)))
            axSel.append(axSum[-1] > thresh)

            # Move Axis to front and index
            self.ten = self.ten.swapaxes(curD, 0)
            self.ten = self.ten[axSel[-1]]
            self.ten = self.ten.swapaxes(0, curD)

            # Build Reconstruction Index
            axInd.append(list(numpy.nonzero(axSel[-1])[0]))
            axInd[-1].append(tenS[curD])

        return (self.ten, axInd)

    def recZero(self, indexList):
        # Note indexList is partially destroyed
        tenS = []
        sliceList = []
        for curI, curList in enumerate(indexList):
            tenS.append(curList.pop(-1))
            sliceList.append(slice(0, ten.shape[curI], 1))
        sliceObj = tuple(sliceList)
        tenR = numpy.zeros(tenS, dtype=self.ten.dtype)
        tenR[sliceObj] = self.ten
        # Now the input tensor resides in upper block of reconstruction tensor

        for curI, curList in enumerate(indexList):
            # Move proper axis to zero
            tenR = tenR.swapaxes(0, curI)
            # Determine list of zero slices
            zeroSlice = list(set(range(tenS[curI])) - set(curList))
            if zeroSlice != []:
                for iS, iR in enumerate(curList):
                    tenR[iR] = tenR[iS]
                tenR[zeroSlice] = 0
            tenR = tenR.swapaxes(0, curI)

        return (tenR)

    def desampleT(self, factor, axis=0):
        if axis != 0:
            data = self.ten.swapaxis(0, axis)
        origShape = list(self.ten.shape)
        newDim = int((origShape[0] - origShape[0] % 3) / 3)
        self.ten = self.ten[:newDim * factor]
        del (origShape[0])
        newShape = [newDim] + [factor] + origShape
        self.ten = numpy.sum(self.ten.reshape(newShape), axis=1)
        if axis != 0:
            self.ten.swapaxis(axis, 0)
        return (self.ten)

    def remove_bad_factors(self, Wall, Hall, ErrTol, features_k):
        sorted_idx = sorted(range(len(ErrTol)), key=lambda k: ErrTol[k])
        to_keep_length = int(np.round(.9 * len(ErrTol)))
        sorted_idx_keep = sorted_idx[:to_keep_length]
        flattened_Wall = Wall.reshape(-1, len(ErrTol))
        flattened_Hall = Hall.reshape(len(ErrTol), -1)
        mod_flattened_Wall = flattened_Wall[:, sorted_idx_keep]
        mod_flattened_Hall = flattened_Hall[sorted_idx_keep, :]
        mod_Wall = mod_flattened_Wall.reshape(-1, len(sorted_idx_keep) * features_k)
        mod_Hall = mod_flattened_Hall.reshape(len(sorted_idx_keep) * features_k, -1)
        mod_ErrTol = ErrTol[sorted_idx_keep]
        return mod_Wall, mod_Hall, mod_ErrTol

    def matSplit(self, name, p_r, p_c, format='npy'):
        if format.lower() == 'npy':
            curMat = numpy.load(name + '.npy')
        else:
            raise ('unknown format')
        try:
            os.mkdir(name)
        except:
            pass

        if curMat.shape[0] % p_r != 0:
            raise ('matrix row dimention not evenly divisible by row processors')
        else:
            rstride = int(curMat.shape[0] / p_r)
        if curMat.shape[1] % p_c != 0:
            raise ('matrix column dimention not evenly divisible by column processors')
        else:
            cstride = int(curMat.shape[1] / p_c)

        indCounter = 0
        for ri in range(p_r):
            for ci in range(p_c):
                outMat = curMat[ri * rstride:(ri + 1) * rstride, ci * cstride:(ci + 1) * cstride]
                numpy.save(name + '/' + name + '_{}.npy'.format(indCounter), outMat)
                indCounter = indCounter + 1

    def primeFactors(self, n):
        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors

    def commonFactors(self, intList):
        factorsList = []
        for curInt in intList:
            factorsList.append(Counter(self.primeFactors(curInt)))
        outCounter = factorsList[0]
        for curI in range(1, len(intList)):
            outCounter = outCounter & factorsList[curI]
        return (list(outCounter.elements()))


class transform_H_index():
    """Collected H factors after MPI operation aren't aligned. This operation performs careful reordering of H factors
    such that the collected factors are aligned"""
    def __init__(self, grid):
        self.p_r = grid[0]
        self.p_c = grid[1]

    def rankidx2blkidx(self):
        """This is to transform the column index to rank index for H"""
        f_idx = []
        for j in range(self.p_c):
            for i in range(self.p_n):
                f_idx.append(i * self.p_n + j)
        return f_idx

    def transform_H_idx(self, rank):
        """This is to transform H based on new index"""
        new_idx_list = self.rankidx2blkidx()
        mod_idx = new_idx_list[rank]
        return mod_idx


def norm(X, comm, norm=2, axis=None, p=-1):
    """Compute the data norm

    Parameters
    ----------
    X : ndarray
       Data to operate on
    comm : object
        MPI communicator object
    norm : int
        type of norm to be computed
    axis : int
        axis of array for the norm to be computed along
    p: int
        Processor count

    Returns
    ----------
    norm : float
        Norm of the given data X
    """
    nm = np.linalg.norm(X, ord=norm, axis=axis) ** 2
    if p != 1:
        nm = comm.allreduce(nm)
    return np.sqrt(nm)


def str2bool(v):
    """Returns instance of string parameter to bool type"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def var_init(clas,var,default):
    """Checks if class attribute is present and if not, intializes the attribute with given default value"""
    if not hasattr(clas,var):
       setattr(clas, var, default)
    return clas.__getattribute__(var)


class parse():
    """Define a class parse which is used for adding attributes """
    def __init__(self):
        pass

class comm_timing(object):
    """
    Decorator class for computing timing for MPI operations. The class uses the global
    variables flag and time initialized in config file and updates them for each call dynamically.

    Parameters
    ----------
    flag: bool
        if Set true, enables the decorator to compute the timings.
    time: dict
        Dictionary to store timing for each function calls
    """
    def __init__(self):
        self.flag = config.flag
        self.time = copy.copy(config.time)

    def __call__(self, original_function):
        if not self.flag: return original_function

        def wrapper_timer(*args, **kwargs):
            start_time = MPI.Wtime()  # 1
            value = original_function(*args, **kwargs)
            end_time = MPI.Wtime()  # 2
            run_time = end_time - start_time  # 3
            self.time[original_function.__name__] = self.time.get(original_function.__name__, 0) + run_time
            config.time.update(self.time)
            return value

        return wrapper_timer
