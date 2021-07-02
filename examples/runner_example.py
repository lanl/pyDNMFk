#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 22:10:42 2021

Run with: mpirun -n 4 python -m runner_example

@author: maksimekineren
"""
from pyDNMFk.runner import pyDNMFk_Runner
import numpy as np

runner = pyDNMFk_Runner(itr=100, init='nnsvd', verbose=True, 
                        norm='fro', method='mu', precision=np.float32,
                        checkpoint=False, sill_thr=0.6)

results = runner.run(grid=[4,1], fpath='../data/', fname='wtsi', ftype='mat', results_path='../results/',
           k_range=[1,3], step_k=1)

W = results["W"]
H = results["H"]