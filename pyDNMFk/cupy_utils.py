# @author: Ismael Boureima

TPB = 32

import cupy as cp
import numpy as np
#from sgemm import sgemm
from .toolz import amber, blue, green, purple, red

# Initializing a pinned memory pool in CuPy to speed up host-device memory transfers.
# However, the following two lines are commented out and thus inactive.
#pinned_memory_pool = cupy.cuda.PinnedMemoryPool()
#cupy.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)
#mem = cp.cuda.alloc_pinned_memory(array.nbytes)

def pin_memory(array):
    """Allocate memory in the pinned (or "page-locked") area. Data in this memory can be transferred to the GPU faster."""
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret

def read_code(code_filename, params):
    """Read a code file and prepend it with CUDA kernel parameter definitions."""
    with open(code_filename, 'r') as f:
        code = f.read()
    for k, v in params.items():
        code = '#define ' + k + ' ' + str(v) + '\n' + code
    return code


def benchmark(func, args, n_run=1):
    """Benchmark a given function with provided arguments over n_run runs. Return a list of execution times."""
    times = []
    for _ in range(n_run):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        func(*args)
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))  # milliseconds
    return times

def timeExecution(func):
    """Decorator to time the execution of a function and print it."""
    def wrapper(*args, **kwargs):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        out = func(*args, **kwargs)
        end.record()
        end.synchronize()
        t = cp.cuda.get_elapsed_time(start, end)  # milliseconds
        print(f"[INFO]: {blue(func.__name__)} ran in {red(round(t,4))} [ms]")
        return out
    return wrapper

@timeExecution
def _asarray(x):
    """Convert the input to a CuPy array."""
    return cp.asarray(x, dtype=x.dtype)

@timeExecution
def _zeros(*args, **kwargs):
    """Return a new CuPy array of given shape and type, filled with zeros."""
    return cp.zeros(*args, **kwargs)

@timeExecution
def _asnumpy(x):
    """Convert a CuPy array to a NumPy array."""
    return cp.asnumpy(x)

@timeExecution
def _divide(a,b):
    """Divide two CuPy arrays element-wise."""
    return cp.divide(a,b, out=None)

@timeExecution
def _matmul(a,b):
    """Matrix multiplication of two CuPy arrays."""
    return cp.matmul(a,b, out=None)

@timeExecution
def _multiply(a,b):
    """Multiply two CuPy arrays element-wise."""
    return cp.multiply(a,b)




