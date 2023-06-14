#!/usr/bin/env python

import os


nS = 5332
nS_final = 5432
number_kpt = 40

os.chdir('5-exciton-Q')
os.system("echo ' ' > xc_%s_%s.dat "%(nS, nS_final))
os.system("echo 'kx ky kz' > kpt_crystal.dat")



for s in range(nS,nS_final+1):
    print(s-nS,'/',nS_final-nS)
    for i in range(1,number_kpt + 1):
        # print(i)
        if os.path.isfile("Q-%s/5.2-absorp-Q/eigenvalues_b1.dat" % i):
            os.system("sed -n %sp Q-%s/5.2-absorp-Q/eigenvalues_b1.dat|awk '{print %s, $1}' >> xc_%s_%s.dat" % (s + 4, i, i,nS,nS_final))

        elif os.path.isfile("Q-%s/5.2-absorp-Q/eigenvalues.dat"%i):
            os.system("sed -n %sp Q-%s/5.2-absorp-Q/eigenvalues.dat|awk '{print %i, $1}' >> xc_%s_%s.dat"%(s+4, i,i, nS,nS_final))
        else:
            print("warning: eigenvalues not found")
            os.system("echo none >> xc_%s_%s.dat "%(nS, nS_final))
        os.system("cat Q-%s/kpt >> kpt_crystal.dat"%i)
    os.system("echo ' ' >> xc_%s_%s.dat " % (nS, nS_final))