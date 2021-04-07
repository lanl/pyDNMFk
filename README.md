# [pyDNMFk: Python Distributed Non Negative Matrix Factorization with determination of hidden features](https://github.com/lanl/pyDNMFk)

[![Build Status](https://github.com/lanl/pyDNMFk/actions/workflows/ci_test.yml/badge.svg?branch=main)](https://github.com/lanl/Distributed_pyNMFk/actions/workflows/ci_test.yml/badge.svg?branch=main) [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg) [![Python Version](https://img.shields.io/badge/python-v3.7.1-blue)](https://img.shields.io/badge/python-v3.7.1-blue)





[pyDNMFk](https://github.com/lanl/pyDNMFk) is a software package for applying non-negative matrix factorization in a distrubuted fashion to large datasets. It has the ability to minimize the difference between reconstructed data and the original data through various norms (Frobenious, KL-divergence).  Additionally, the Custom Clustering algorithm allows for automated determination for the number of Latent features 

<hr/>

## Features:

* Utilization of MPI4py for distributed operation.
* Distributed NNSVD and SVD initiaizations.
* Distributed Custom Clustering algorithm for estimating automated latent feature number (k) determination.
* Objective of minimization of KL divergence/Frobenius norm. 
* Optimization with multiplicative updates, BCD, BPP and HALS. 

## Installation:

```
git clone https://github.com/lanl/pyDNMFk.git
cd pynmfk
conda create --name pyDNMFk python=3.7.1 mpi4py
source activate pyDNMFk
python setup.py install
```

## Prerequisites:
* conda
* numpy>=1.2
* matplotlib
* MPI4py
* scipy
* h5py

## Usage
We provide a sample dataset that can be used for estimation of k:
```python
'''Imports block'''

import sys
import pyDNMFk.config as config
config.init(0)
from pyDNMFk.pyDNMFk import *
from pyDNMFk.data_io import *
from pyDNMFk.dist_comm import *
from scipy.io import loadmat
from mpi4py import MPI
comm = MPI.COMM_WORLD
args = parse()  


'''parameters initialization block'''


# Data Read here
args.fpath = 'data/'
args.fname = 'wtsi'  
args.ftype = 'mat'
args.precision = np.float32

#Distributed Comm config block
p_r, p_c = 4, 1  

#NMF config block
args.norm = 'kl'
args.method = 'mu'
args.init = 'nnsvd'
args.itr = 5000
args.verbose = True

#Cluster config block
args.start_k = 2 
args.end_k = 5
args.sill_thr = 0.9

#Data Write
args.results_path = 'results/'


'''Parameters prep block'''


comms = MPI_comm(comm, p_r, p_c)
comm1 = comms.comm
rank = comm.rank
size = comm.size
args.size, args.rank, args.comm, args.p_r, args.p_c = size, rank, comms, p_r, p_c
args.row_comm, args.col_comm, args.comm1 = comms.cart_1d_row(), comms.cart_1d_column(), comm1
A_ij = data_read(args).read().astype(args.precision)

nopt = PyNMFk(A_ij, factors=None, params=args).fit()
print('Estimated k with NMFk is ',nopt)
```
See the resources for more use cases.
<hr/>

## Authors:

* [Manish Bhattarai](mailto:ceodspspectrum@lanl.gov) - Los Alamos National Laboratory
* [Ben Nebgen](mailto:bnebgen@lanl.gov) - Los Alamos National Laboratory
* [Erik Skau](mailto:ewskau@lanl.gov) - Los Alamos Natioanl Laboratory
* [Boian Alexandrov](mailto:boian@lanl.gov) - Los Alamos Natioanl Laboratory

## Citation:
```@inproceedings{bhattarai2020distributed,
  title={Distributed Non-Negative Tensor Train Decomposition},
  author={Bhattarai, Manish and Chennupati, Gopinath and Skau, Erik and Vangara, Raviteja and Djidjev, Hristo and Alexandrov, Boian S},
  booktitle={2020 IEEE High Performance Extreme Computing Conference (HPEC)},
  pages={1--10},
  year={2020},
  organization={IEEE}
}
```
```@article{chennupati2020distributed,
  title={Distributed non-negative matrix factorization with determination of the number of latent features},
  author={Chennupati, Gopinath and Vangara, Raviteja and Skau, Erik and Djidjev, Hristo and Alexandrov, Boian},
  journal={The Journal of Supercomputing},
  pages={1--31},
  year={2020},
  publisher={Springer}
}
```
## Acknowledgments:
Los Alamos National Lab (LANL), T-1

## Copyright Notice:

© (or copyright) 2020. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.

## License:

This program is open source under the BSD-3 License.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



