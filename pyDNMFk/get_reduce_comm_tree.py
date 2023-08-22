import numpy as np
def getReduceCommTree(x, VRBZ=False):
    """
    Generate a reduction communication tree for the given array.

    This function computes the reduction communication tree structure for a given array x. 
    It constructs the communication hierarchy in multiple levels such that in each level, 
    some elements are designated as "workers" and others as "roots". Workers send data to roots. 
    The process continues recursively on the roots until a single root is left.

    Parameters:
    -----------
    x : array-like
        The input array for which the reduction communication tree is to be generated.

    VRBZ : bool, optional
        Verbose mode flag. If set to True, the function prints the intermediate computation details.
        Default is False.

    Returns:
    --------
    reduceCommTree : dict
        A dictionary containing the communication hierarchy in various levels. 
        Each level is represented as a key 'l{i}' where {i} is the level number.
        For each level, the dictionary contains:
        - 'roots'   : array of elements that act as data receivers in that level.
        - 'workers' : array of elements that send data to the roots in that level.
        - 'idx'    : array of indices mapping the original array elements to their index 
                     in the current 'roots' and 'workers' arrays.

    Notes:
    ------
    If the length of the input array is odd, an additional communication is added at the end, 
    where the last element sends its data to the first element.

    Example:
    --------
    #>>> x = np.array([0, 1, 2, 3, 4])
    #>>> getReduceCommTree(x)
    {'l1': {'roots': array([0, 2, 4]), 
            'workers': array([1, 3]), 
            'idx': array([ 0,  0,  1,  1,  2], dtype=int32)},
     'l2': {'roots': array([0]), 
            'workers': array([4]), 
            'idx': array([ 0, -1, -1, -1,  0], dtype=int32)},
     'l3': {'roots': [0], 
            'workers': [4], 
            'idx': array([ 0, -1, -1, -1,  0], dtype=int32)}}
    """
    IDX = np.zeros(len(x), dtype=np.int32) -1
    i = 1
    x0 = np.copy(x)
    n = len(x0)
    UNEVEN = False
    if n%2 != 0:
        UNEVEN = True
        x0 = x0[:-1]
    reduceCommTree = {}
    while n > 1:
        s0                      = 2**(i-1)
        s                       = 2**(i)
        y                       = np.int32(x0%s)
        roots                   = x0[np.where(y == 0)]
        workers                 = x0[np.where(y == s0)]
        reduceCommTree[f'l{i}'] = {'roots':roots, 'workers':workers}
        idx = IDX +0
        for k in x:
            if k in roots:
                idx[k] = np.where(roots == k)[0][0]
            elif k in workers:
                idx[k] = np.where(workers == k)[0][0]
            else:
                pass
        reduceCommTree[f'l{i}']['idx'] = idx
        if VRBZ:
            print(f"################### [i={i}|s={s}|n={n}] #################")
            #print(f"   x0      = {x0}")
            #print(f"   y       = {y}")
            print(f"   roots   = {roots}")
            print(f"   workers = {workers}")
            #for j in range(len(roots)): print(f"{workers[j]} --> {roots[j]}")
        x0 = roots
        n =len(x0)
        i+=1
    if UNEVEN:
        if VRBZ: print(f"{x[-1]} --> {x[0]}")
        idx =  IDX +0
        idx[0], idx[-1] = 0,0
        reduceCommTree[f'l{i}'] = {'roots':[0], 'workers':[x[-1]], 'idx':idx}
    return reduceCommTree

