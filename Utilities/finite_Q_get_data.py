#!/usr/bin/env python

import os
import sys

if len(sys.argv) < 3:
    print("Usage: %s nS n_Qpt"%(sys.argv[0]))
    print("nS is the index of exciton bands; n_Qpt is number of n-Q")
    sys.exit(1)

nS = int(sys.argv[1])
number_kpt = int(sys.argv[2])

# nS = 1
# number_kpt = 576

os.chdir('5-exciton-Q')
os.system("echo 'eigenvalues' > xc_%s.dat "%nS)
os.system("echo 'kx ky kz' > kpt_crystal.dat")
for i in range(1,number_kpt + 1):
    print(i)
    if i == 1:
        if os.path.isfile("Q-%s/5.2-absorp-Q/eigenvalues_b1.dat" % i):
            os.system("sed -n %sp Q-%s/5.2-absorp-Q/eigenvalues_b1.dat|awk '{print $1}' >> xc_%s.dat" % (nS + 4, i, nS))
        else:
            print("warning: eigenvalues not found")
            os.system("echo none >> xc_%s.dat" % (nS))
        os.system("cat Q-%s/kpt >> kpt_crystal.dat" % i)

    else:
        if os.path.isfile("Q-%s/5.2-absorp-Q/eigenvalues.dat"%i):
            os.system("sed -n %sp Q-%s/5.2-absorp-Q/eigenvalues.dat|awk '{print $1}' >> xc_%s.dat"%(nS+4, i, nS))
        else:
            print("warning: eigenvalues not found")
            os.system("echo none >> xc_%s.dat" % (nS))
        os.system("cat Q-%s/kpt >> kpt_crystal.dat"%i)