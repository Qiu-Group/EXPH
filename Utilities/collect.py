#!/usr/bin/env python

import os
import sys
# TODO!!
#for i in {1..3};do cp Q-$i/5.2-absorp-Q/eigenvectors.h5 save/eigenvectors_$i.h5;done
if len(sys.argv) < 3:
    print("Usage: %s nc nv"%(sys.argv[0]))
    print("nc(nv) is number of conduction(valence) bands in kernel.cplx.x")
    sys.exit(1)

nc = sys.argv[1]
nv = sys.argv[2]