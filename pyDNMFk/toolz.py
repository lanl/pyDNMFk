import numpy as np
import numpy # Just for type checking
import os, sys, gc



#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\  DEFINING PRINTING OPERATIONS:
# Defining some printing color schemes:_
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''

def amber(msg):
    return bcolors.WARNING + str(msg) + bcolors.ENDC
def blue(msg):
    return bcolors.OKBLUE  + str(msg) + bcolors.ENDC
def green(msg):
    return bcolors.OKGREEN + str(msg) + bcolors.ENDC
def purple(msg):
    return bcolors.HEADER + str(msg) + bcolors.ENDC
def red(msg):
    return bcolors.FAIL + str(msg) + bcolors.ENDC

def print_there(x, y, text):
    sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (x, y, text))
    sys.stdout.flush()
def printRC(x, y, text):
    sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (x, y, text))
    sys.stdout.flush()
def printXY( y,x, text):
    sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (x, y, text))
    sys.stdout.flush()
def getTerminalSize():
    """
    Get Terminal size (nLinesxnColums)
    returns nLines, nColums

    """
    env = os.environ
    def ioctl_GWINSZ(fd):
        try:
            import fcntl, termios, struct, os
            cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ,
        '1234'))
        except:
            return
        return cr
    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except:
            pass
    if not cr:
        cr = (env.get('LINES', 25), env.get('COLUMNS', 80))

        ### Use get(key[, default]) instead of a try/catch
        #try:
        #    cr = (env['LINES'], env['COLUMNS'])
        #except:
        #    cr = (25, 80)
    return int(cr[0]), int(cr[1])



def log(msg, rank=0, lrank=0):
     print(f"({green('G%02d'%rank)}|{blue('L%02d'%lrank)})> {msg}")  #%(rank, lrank))
     sys.stdout.flush()

def lrLog(msg, rank=0, lrank=0):
    if LMASTER: log(msg=msg, rank=rank, lrank=lrank)
def grLog(msg, rank=0, lrank=0):
    if GMASTER: log(msg=msg, rank=rank, lrank=lrank)




def get_index(val,Array):
    """
    Function used to find index of closest value to a traget value inside an Array
    val   : target value to find in the Array
    Array : Array being investigated
    idx   : Location of target value in Array
    """
    temp = np.abs(Array-val)  # Replace each element of temp array by the absolute value of (that element - 1)
    idx = temp.argmin()     # Look for the minimum value in resulting temp
    #print 'target located at :', idx
    return idx


def get_loc(val,x,fx,debug=False):
    """
    Calculates the x-coordinate (xp) for a given target value, such that ``f(xp) = val``,
    given the arrays ``x`` and ``f(x)``. 

    This function employs a linear interpolation between two nearest points in the 
    ``f(x)`` array to determine the x-coordinate for the given target value.
    Graphically, the logic follows the concept illustrated in the ASCII diagram
    provided in the function.

    Parameters
    ----------
    val : float
        The target value for which the x-coordinate is to be determined.
    
    x : list or numpy.array
        The x-coordinates array.
        
    fx : list or numpy.array
        Array containing the values of the function f evaluated at each x.
    
    debug : bool, optional
        If True, prints debug messages. Defaults to False.

    Returns
    -------
    float
        The interpolated x-coordinate (xp) corresponding to the provided target value.

    Raises
    ------
    Exception
        If unable to locate the given target value within the provided range.
    """  

    idx = get_index(val,fx)
    if debug: print("[+] found closest point @ idx = {}".format(idx))
    if fx[idx] == val:
        if debug:print("[+] point is actually exact")
        return x[idx]
    elif fx[idx] > val:
        if debug:print("[+] point ahead")
        i0, i1 = idx-1,idx
    elif fx[idx] < val:
        if debug:print("[+] point behind")
        i0, i1 = idx, idx+1
    else:
        print('[!] ERROR in get_loc(val,x,fx): Unable to locate {}'.format(val))
        return
    x0, x1 = x[i0],x[i1]
    f0, f1 = fx[i0],fx[i1]
    return x0 + (val-f0)*(x1-x0)/(f1-f0)



