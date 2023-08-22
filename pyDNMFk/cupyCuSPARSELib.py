# @author: Ismael Boureima
import cupy, cupyx,  numpy
import scipy
from .toolz import amber, blue, green, red
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, dia_matrix, issparse

def spMat(A, Format='coo', shape=None, dtype=None, copy=False):
    r"""
    Converts a given matrix A into a specified sparse format using CuPy.

    Parameters:
    -----------
    A : ndarray or sparse matrix
        Input matrix to be converted.
    Format : {'coo', 'csr', 'csc', 'dia'}
        Desired sparse format. Default is 'coo'.
    shape : tuple, optional
        Desired shape for the sparse matrix.
    dtype : data-type, optional
        Data type of the result.
    copy : bool, optional
        If true, it guarantees the input data A is not modified. Default is False.

    Returns:
    --------
    Sparse matrix in the desired format and library (cupy).
    """
    fmt = Format.lower()
    assert fmt in ['coo', 'csr', 'csc', 'dia'], f"[{red('!')}] Format must be in [{green('coo')}, {green('crs')}, {green('csc')}, {green('dia')}] "
    isNdArr = False
    if type(A) in [numpy.ndarray]: isNdArr=True
    if fmt in ['coo']:
        if isNdArr:
            return cupyx.scipy.sparse.coo_matrix(coo_matrix(A), shape=shape, dtype=dtype, copy=copy)
        else: 
            return cupyx.scipy.sparse.coo_matrix(A, shape=shape, dtype=dtype, copy=copy)
    elif fmt in ['csc']:
        if isNdArr:
            return cupyx.scipy.sparse.csc_matrix(coo_matrix(A), shape=shape, dtype=dtype, copy=copy)
        else:
            return cupyx.scipy.sparse.csc_matrix(A, shape=shape, dtype=dtype, copy=copy)
    elif fmt in ['csr']:
        if isNdArr:
            return cupyx.scipy.sparse.csr_matrix(coo_matrix(A), shape=shape, dtype=dtype, copy=copy)
        else:
            return cupyx.scipy.sparse.csr_matrix(A, shape=shape, dtype=dtype, copy=copy)
    elif fmt in ['dia']:
        if isNdArr:
            return cupyx.scipy.sparse.dia_matrix(coo_matrix(A), shape=shape, dtype=dtype, copy=copy)
        else:
            return cupyx.scipy.sparse.dia_matrix(A, shape=shape, dtype=dtype, copy=copy)
    else:
        print(f"[!] sparse matrix format '{fmt}' is not supported")



def spCu2Sc(A, copy=False):
    r"""
    Converts a CuPy sparse matrix to a SciPy sparse matrix.

    Parameters:
    -----------
    A : cupyx sparse matrix
        Input matrix to be converted.
    copy : bool, optional
        If true, it guarantees the input data A is not modified. Default is False.

    Returns:
    --------
    Sparse matrix in the desired format and library (scipy).
    """
    fmt = A.format.lower()
    assert fmt in ['coo', 'csr', 'csc', 'dia'], f"[{red('!')}] Format must be in [{green('coo')}, {green('crs')}, {green('csc')}, {green('dia')}] "
    if fmt in ['coo']:
        return scipy.sparse.coo_matrix((A.data.get(),A.indices.get(),A.indptr.get()),shape=A.shape,dtype=A.dtype)
    elif fmt in ['csc']:
        return scipy.sparse.csc_matrix((A.data.get(),A.indices.get(),A.indptr.get()),shape=A.shape,dtype=A.dtype)
    elif fmt in ['csr']:
        return scipy.sparse.csr_matrix((A.data.get(),A.indices.get(),A.indptr.get()),shape=A.shape,dtype=A.dtype)
    elif fmt in ['dia']:
        return scipy.sparse.dia_matrix((A.data.get(),A.indices.get(),A.indptr.get()),shape=A.shape,dtype=A.dtype)
    else:
        print(f"[!] sparse matrix format '{fmt}' is not supported")

def spSc2Cu(A, copy=False):
    r"""
    Converts a SciPy sparse matrix to a CuPy sparse matrix.

    Parameters:
    -----------
    A : scipy sparse matrix
        Input matrix to be converted.
    copy : bool, optional
        If true, it guarantees the input data A is not modified. Default is False.

    Returns:
    --------
    Sparse matrix in the desired format and library (cupy).
    """
    fmt = A.format.lower()
    assert fmt in ['coo', 'csr', 'csc', 'dia'], f"[{red('!')}] Format must be in [{green('coo')}, {green('crs')}, {green('csc')}, {green('dia')}] "
    if fmt in ['coo']:
        return cupyx.scipy.sparse.coo_matrix((cp.array(A.data), cp.array(A.indices), cp.array(A.indptr)), shape=A.shape, dtype=A.dtype)
    elif fmt in ['csc']:
        return cupyx.scipy.sparse.csc_matrix((cp.array(A.data), cp.array(A.indices), cp.array(A.indptr)), shape=A.shape, dtype=A.dtype)
    elif fmt in ['csr']:
        return cupyx.scipy.sparse.csr_matrix((cp.array(A.data), cp.array(A.indices), cp.array(A.indptr)), shape=A.shape, dtype=A.dtype)
    elif fmt in ['dia']:
        return cupyx.scipy.sparse.dia_matrix((cp.array(A.data), cp.array(A.indices), cp.array(A.indptr)), shape=A.shape, dtype=A.dtype)
    else:
        print(f"[!] sparse matrix format '{fmt}' is not supported")


def spRandMat(nr, nc, density, Format='coo', dtype=cupy.float32):
    r"""
    Generates a random sparse matrix using CuPy.

    Parameters:
    -----------
    nr : int
        Number of rows.
    nc : int
        Number of columns.
    density : float
        Desired density for the sparse matrix. Values between 0 and 1.
    Format : {'coo', 'csr', 'csc', 'dia'}, optional
        Desired sparse format. Default is 'coo'.
    dtype : data-type, optional
        Data type of the result. Default is cupy.float32.

    Returns:
    --------
    Random sparse matrix in the desired format and library (cupy).
    """
    fmt = Format.lower()
    return cupyx.scipy.sparse.random(nr, nc, density=density, format=fmt, dtype=dtype)


def spMM(A, B, alpha=1.0, beta=0.0, transpA=False, transpB=False):
    r"""
    Performs matrix multiplication of sparse matrix A and dense matrix B using CuPy.

    Parameters:
    -----------
    A : cupyx sparse matrix
        Left sparse matrix.
    B : ndarray
        Right dense matrix.
    alpha : float, optional
        Scalar multiplier for the product of A and B. Default is 1.0.
    beta : float, optional
        Scalar multiplier for the initial matrix C (if provided). Default is 0.0.
    transpA : bool, optional
        If True, transpose matrix A before multiplication. Default is False.
    transpB : bool, optional
        If True, transpose matrix B before multiplication. Default is False.

    Returns:
    --------
    Resultant matrix after multiplication.
    """
    assert cupy.cusparse.check_availability('spmm'), "[!] spmm is not available"
    #if transpA : A = A.T
    #if transpB : B = B.T
    if not A.has_canonical_format: A.sum_duplicates()
    B = cupy.array(B, order='f')
    return cupy.cusparse.spmm( A, B, alpha=alpha, beta=beta, transa=transpA, transb=transpB)
    

def spMM_with_C(A, B, C , alpha=1.0, beta=0.0, transpA=False, transpB=False):
    r"""
    Performs matrix multiplication of sparse matrix A and dense matrix B, and adds it to matrix C using CuPy.

    Parameters:
    -----------
    A : cupyx sparse matrix
        Left sparse matrix.
    B : ndarray
        Right dense matrix.
    C : ndarray
        Dense matrix to which the result is added.
    alpha : float, optional
        Scalar multiplier for the product of A and B. Default is 1.0.
    beta : float, optional
        Scalar multiplier for the initial matrix C. Default is 0.0.
    transpA : bool, optional
        If True, transpose matrix A before multiplication. Default is False.
    transpB : bool, optional
        If True, transpose matrix B before multiplication. Default is False.

    Returns:
    --------
    Resultant matrix after multiplication and addition to C.
    """
    assert cupy.cusparse.check_availability('spmm'), "[!] spmm is not available"
    #if transpA : A = A.T
    #if transpB : B = B.T
    if not A.has_canonical_format: A.sum_duplicates()
    B = cupy.array(B, order='f')
    C = cupy.array(C, order='f')
    return cupy.cusparse.spmm( A, B, c=C, alpha=alpha, beta=beta, transa=transpA, transb=transpB)

