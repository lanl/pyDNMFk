# @author: Manish Bhattarai, Ben Nebgen, Erik Skau
import copy
from collections import Counter

from . import config
config.init(0)
import numpy
import numpy as np
from mpi4py import MPI
import pickle
from sklearn.preprocessing import LabelBinarizer
import json
from sklearn.neural_network import MLPClassifier

class determine_block_params():
    r"""Computes the parameters  for each chunk to be read by MPI process

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
        self.rank = self.rank if np.product(self.pgrid)>1 else 0
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
    r"""Performs various operations on the data

    Parameters
    ----------
    data : ndarray
       Data to operate on"""
    def __init__(self, data,params):
        self.ten = data
        self.params = params
        self.comm1 = self.params.comm1
        self.cart_1d_row = self.params.row_comm
        self.cart_1d_column = self.params.col_comm
        self.rank = self.comm1.rank
        self.p_r = self.params.p_r
        self.p_c = self.params.p_c
        self.topo  = self.params.topo
        self.k = self.params.k
        self.compute_global_dim()
        self.compute_local_dim()
        (self.A_ij_m,self.A_ij_n) = self.ten.shape
        self.m = self.params.m
        self.n = self.params.n

    def compute_global_dim(self):
        """Computes global dimensions m and n from given chunk sizes for any grid configuration"""
        self.loc_m, self.loc_n = self.ten.shape
        if self.p_r != 1 and self.p_c == 1:
            self.params.n = self.loc_n
            self.params.m = self.comm1.allreduce(self.loc_m)
        elif self.p_c != 1 and self.p_r == 1:
            self.params.n = self.comm1.allreduce(self.loc_n)
            self.params.m = self.loc_m
        else:
            if self.rank % self.p_c == 0:
                self.params.m = self.loc_m
            else:
                self.params.m = 0
            self.params.m = self.comm1.allreduce(self.params.m)
            if self.rank // self.p_c == 0:
                self.params.n = self.loc_n
            else:
                self.params.n = 0
            self.params.n = self.comm1.allreduce(self.params.n)
            self.comm1.barrier()

        # if self.rank == 0: print('Data dimensions=(', self.params.m, self.params.n, ')')

    def compute_local_dim(self):
        r"""Computes local dimensions for factors from given chunk sizes for any grid configuration"""
        if self.topo == '2d':
            dtr_blk_m = determine_block_params(self.cart_1d_column, (self.p_c, 1), (self.ten.shape[0], self.k))
            m_loc = dtr_blk_m.determine_block_shape_asymm()[0]
            dtr_blk_n = determine_block_params(self.cart_1d_row, (1, self.p_r), (self.k, self.ten.shape[1]))
            n_loc = dtr_blk_n.determine_block_shape_asymm()[1]
        elif self.topo== '1d':
            dtr_blk_m = determine_block_params(self.comm1, (self.p_r, 1), (self.params.m, self.k))
            m_loc = dtr_blk_m.determine_block_shape_asymm()[0]
            dtr_blk_n = determine_block_params(self.comm1, (1, self.p_c), (self.k, self.params.n))
            n_loc = dtr_blk_n.determine_block_shape_asymm()[1]
        w_idx_range = dtr_blk_m.determine_block_index_range_asymm()
        h_idx_range = dtr_blk_n.determine_block_index_range_asymm()
        w_start,w_end = w_idx_range[0][0],w_idx_range[1][0]+1
        h_start,h_end = h_idx_range[0][1], h_idx_range[1][1] + 1
        self.params.m_loc,self.params.n_loc = m_loc,n_loc
        self.params.W_start,self.params.W_end = w_start,w_end
        self.params.H_start,self.params.H_end = h_start,h_end

    def zero_idx_prune(self):
        r"""Computes the row and columns indices of the data matrix to be pruned"""
        row_sum = np.sum(self.ten != 0, 1)
        col_sum = np.sum(self.ten != 0, 0)
        if self.topo=='2d':
            row_sum = self.cart_1d_column.allreduce(row_sum)
            col_sum = self.cart_1d_row.allreduce(col_sum)
        else:
            if self.p_c>1:row_sum = self.comm1.allreduce(row_sum)
            if self.p_r>1:col_sum = self.comm1.allreduce(col_sum)
        row_zero_idx_x = row_sum > 0
        col_zero_idx_x = col_sum > 0
        if self.topo == '2d':
            col_zero_idx_h = col_sum[self.params.H_start:self.params.H_end] > 0
            row_zero_idx_w = row_sum[self.params.W_start:self.params.W_end] > 0
        elif self.topo == '1d':
            row_zero_idx_w = row_sum > 0
            col_zero_idx_h = col_sum > 0
        return row_zero_idx_x,col_zero_idx_x,row_zero_idx_w,col_zero_idx_h

    def prune(self,data,row_zero_idx,col_zero_idx):
        """Performs pruning of data

        Parameters
        ----------
        data : ndarray
            data to be pruned
        row_zero_idx : list
            indices comprising zero/non-zero rows
        col_zero_idx : list
            indices comprising zero/non-zero columns

        Returns
        -------
        data : ndarray
            Pruned data

        """
        data = data[np.ix_(row_zero_idx, col_zero_idx)]
        return data

    def prune_all(self,W,H):
        """ Prunes data and factors

        Parameters
        ----------
        W : ndarray
        H : ndarray

        Returns
        -------
        X : ndarray
        W : ndarray
        H : ndarray
        """
        self.params.row_zero_idx_x,self.params.col_zero_idx_x,self.params.row_zero_idx_w,self.params.col_zero_idx_h = self.zero_idx_prune()
        self.ten = self.prune(self.ten,self.params.row_zero_idx_x,self.params.col_zero_idx_x)
        W = self.prune(W,self.params.row_zero_idx_w,[True]*W.shape[1])
        H = self.prune(H,[True]*H.shape[0],self.params.col_zero_idx_h)
        return self.ten,W,H

    def unprune(self,data,row_zero_idx,col_zero_idx):
        """ Unprunes data

        Parameters
        ----------
        data : ndarray
            Data to be unpruned
        row_zero_idx : list
            indices comprising zero/non-zero rows
        col_zero_idx : list
            indices comprising zero/non-zero cols

        Returns
        -------

        """
        if len(row_zero_idx)>1:
            B = np.zeros((len(row_zero_idx),data.shape[1]))
            B[row_zero_idx,:] = data
        elif len(col_zero_idx)>1:
            B = np.zeros((data.shape[0],len(col_zero_idx)))
            B[:,col_zero_idx] = data
        return B

    def unprune_factors(self,W,H):
        """ Unprunes the factors

        Parameters
        ----------
        W : ndarray
        H : ndarray

        Returns
        -------
        W : ndarray
        H : ndarray
        """
        W = self.unprune(W,self.params.row_zero_idx_w,[])
        H = self.unprune(H,[],self.params.col_zero_idx_h)
        return W,H



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
            for i in range(self.p_r):
                f_idx.append(i * self.p_r + j)
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

class serialize_deserialize_mlp():
    """Returns model/parameters of the model in dictionary format"""
    def __init__(self,model_name = None,model=None):
        self.model_name = model_name
        self.model = model

    def from_json(self):
        """Load the model from JSON"""
        with open(self.model_name, 'r') as model_json:
            model_dict = json.load(model_json)
            self.model = self.deserialize(model_dict)
        return self.model

    def to_json(self):
        """Write the model paramters to JSON"""
        with open(self.model_name, 'w') as model_json:
            json.dump(self.serialize(), model_json)

    def serialize(self):
        """Convert the model  into a a dictionary of parameters"""
        serialized_label_binarizer = {
            'neg_label':self.model._label_binarizer.neg_label,
            'pos_label':self.model._label_binarizer.pos_label,
            'sparse_output':self.model._label_binarizer.sparse_output,
            'y_type_':self.model._label_binarizer.y_type_,
            'sparse_input_':self.model._label_binarizer.sparse_input_,
            'classes_':self.model._label_binarizer.classes_.tolist()
        }
        serialized_model = {
            'meta': 'mlp',
            'coefs_': [array.tolist() for array in self.model.coefs_],
            'loss_': self.model.loss_,
            'intercepts_': [array.tolist() for array in self.model.intercepts_],
            'n_iter_': self.model.n_iter_,
            'n_layers_': self.model.n_layers_,
            'n_outputs_': self.model.n_outputs_,
            'out_activation_': self.model.out_activation_,
            '_label_binarizer': serialized_label_binarizer,
            'params': self.model.get_params()
        }
        if isinstance(self.model.classes_, list):
            serialized_model['classes_'] = [array.tolist() for array in self.model.classes_]
        else:
            serialized_model['classes_'] = self.model.classes_.tolist()
        return serialized_model


    def deserialize(self,model_dict):
        """Convert the dictionary of parameters into model"""
        model = MLPClassifier(**model_dict['params'])
        model.coefs_ = numpy.array([numpy.array(i) for i in model_dict['coefs_']],dtype='object')
        model.loss_ = model_dict['loss_']
        model.intercepts_ = numpy.array([numpy.array(i) for i in model_dict['intercepts_']],dtype='object')
        model.n_iter_ = model_dict['n_iter_']
        model.n_layers_ = model_dict['n_layers_']
        model.n_outputs_ = model_dict['n_outputs_']
        model.out_activation_ = model_dict['out_activation_']
        label_binarizer = LabelBinarizer()
        label_binarizer_dict = model_dict['_label_binarizer']
        label_binarizer.neg_label = label_binarizer_dict['neg_label']
        label_binarizer.pos_label = label_binarizer_dict['pos_label']
        label_binarizer.sparse_output = label_binarizer_dict['sparse_output']
        label_binarizer.y_type_ = label_binarizer_dict['y_type_']
        label_binarizer.sparse_input_ = label_binarizer_dict['sparse_input_']
        label_binarizer.classes_ = numpy.array(label_binarizer_dict['classes_'])
        model._label_binarizer = label_binarizer
        model.classes_ = numpy.array(model_dict['classes_'])
        return model

def str2bool(v):
    """Returns instance of string parameter to bool type"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise NameError('Boolean value expected.')

def var_init(clas,var,default):
    """Checks if class attribute is present and if not, intializes the attribute with given default value"""
    if not hasattr(clas,var):
       setattr(clas, var, default)
    return clas.__getattribute__(var)


class parse():
    """Define a class parse which is used for adding attributes """
    def __init__(self):
        pass


class Checkpoint():

    def __init__(self, checkpoint_save, params):
        """
        Class to demo checkpoint saving and continuing from checkpoint

        Parameters
        ----------
        checkpoint_save : bool, optional
            Enable/disable checkpoint
        max_iters : TYPE, optional
            maximum number of iterations (epoch). The default is 50.

        Returns
        -------
        None.

        """
        self.checkpoint_save = checkpoint_save if checkpoint_save else False
        self.params = params
        self.perturbation = 0
        self.k = 0
        self.flag = 0


    def load_from_checkpoint(self):
        """run from checkpoint instead"""
        if self.checkpoint_save:
            saved_class = pickle.load(open(self.params.results_path+"/checkpoint.p", "rb"))
            if self.params.rank == 0:print("Checkpoint loaded")
            # copy the saved state to the class
            if self.params.rank == 0:print("Loading saved object state...")
            self._set_params(vars(saved_class))
            if self.params.rank == 0:print("Continuing from checkpoint for k=", self.k,'perturbation=',self.perturbation)


    def _save_checkpoint(self,flag,perturbation,k):
        args = parse()
        """Saves the class at checkpoint number"""
        args.flag = flag
        args.perturbation = perturbation
        args.k = k
        # save the class object
        if self.checkpoint_save and self.params.rank==0:
           pickle.dump(args, open(self.params.results_path+"checkpoint.p", "wb"))
           print('checkpoint saved')

    def _set_params(self, class_parameters):
        """Sets class variables from the loaded checkpoint"""
        for parameter, value in class_parameters.items():
            setattr(self, parameter, value)


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
