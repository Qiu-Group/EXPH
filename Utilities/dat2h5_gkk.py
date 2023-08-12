#!/usr/bin/env python

import numpy as np
import h5py as h5
import sys

if len(sys.argv) != 2:
    print("Usage[elphmat.dat/elphmat_phase.dat]: p/np")
    sys.exit(1)

transfer = None
if sys.argv[1] == "np": transfer = 0
if sys.argv[1] == "p": transfer =1
if transfer == None: raise Exception("input np or p")

if transfer == 0:
    data = np.loadtxt("elphmat.dat")
    h5_elphmat = h5.File('elphmat.h5','w')
    h5_elphmat.create_dataset('data', data=data)
    h5_elphmat.close()

if transfer == 1:
    data = np.loadtxt("elphmat_phase.dat")
    h5_elphmat = h5.File('elphmat_phase.h5','w')
    h5_elphmat.create_dataset('data', data=data)
    h5_elphmat.close()