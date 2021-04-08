Welcome to pyDNMFk's documentation!
===================================

pyDNMFk is a software package for applying non-negative matrix factorization in a distrubuted fashion to large datasets. It has the ability to minimize the difference between reconstructed data and the original data through various norms (Frobenious, KL-divergence). Additionally, the Custom Clustering algorithm allows for automated determination for the number of Latent features.

Features
========================

* Utilization of MPI4py for distributed operation.
* Distributed NNSVD and SVD initiaizations.
* Distributed Custom Clustering algorithm for estimating automated latent feature number (k) determination.
* Objective of minimization of KL divergence/Frobenius norm.
* Optimization with multiplicative updates, BCD, and HALS.


Scalability
========================
pyDNMFk Scales from laptops to clusters. The library is convenient on a laptop. It can be installed easily  with conda or pip and extends the matrix decomposition from a single core to numerous cores across nodes.
pyDNMFk is efficient and has been tested on powerful servers across LANL and Oakridge scaling beyond 1000+ nodes.
This library facilitates the transition between single-machine to large scale cluster so as to enable users to both start simple and scale up when necessary.


Installation
========================

.. code-block:: console

   git clone https://github.com/lanl/pyDNMFk.git
   cd pyDNMFk
   conda create --name pyDNMFk python=3.7.1 openmpi mpi4py
   source activate pyDNMFk
   python setup.py install


Usage Example
========================
We provide a sample dataset that can be used for estimation of k:

.. code-block:: python

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


Indices and tables
========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules


Indices and tables
========================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`















