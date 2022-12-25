#!/usr/bin/env python

import os
import sys
# TODO!!
#for i in {1..3};do cp Q-$i/5.2-absorp-Q/eigenvectors.h5 save/eigenvectors_$i.h5;done
if len(sys.argv) != 3:
    print("Usage[acvs]: acv nQ")
    print("Usage[gkk]: gkk n_lambda")
    print("nQ: number of n-Q\nn_lambda: number of phonon mode")
    sys.exit(1)

second_parameter = sys.argv[2]

if sys.argv[1] == 'acv':
    os.mkdir('./acvs.save')
    os.system('for i in {1..%s};do cp Q-$i/5.2-absorp-Q/eigenvectors.h5 acvs.save/eigenvectors_$i.h5;done'%second_parameter)
    print('data collected!')

elif sys.argv[1] == 'gkk':
    os.mkdir('./gkk.save')
    os.system('cp epw.out gkk.save')
    os.chdir('./gkk.save')

    os.system("cat epw.out | grep -E '^ +[0-9]+ +[0-9]+ +[0-9]+'|awk '{print $7}' > elphmat.dat")
    os.system("grep '     iq' epw.out |awk '{print $5,$6,$7}' > q.dat")
    os.system("grep 'lattice parameter ' epw.out |awk '{print $5}' >a0.dat")
    os.system("grep '            a' epw.out|awk '{print $4,$5,$6}' > a.dat")
    os.system("grep '            b' epw.out|awk '{print $4,$5,$6}' > b.dat")
    os.system("head epw.out -n 5000|grep 'Using uniform q-mesh:' |awk '{print $4,$5,$6}' > mesh_temp")

    temp = open('mesh_temp','r')
    mesh = temp.readline().strip()
    temp.close()
    os.system('rm mesh_temp')

    os.system("kmesh.pl %s |grep '^ '|awk '{print $1,$2,$3}' > k.dat"%mesh)
    os.system("grep 'ik =       1' epw.out -A %s| grep -E '^ +[0-9]+ +[0-9]+ +[0-9]+'|awk '{print $6}' > omega.dat"%(int(second_parameter) + 2))


else:
    raise Exception("the first input can only be acv or gkk; acv for acvs.save and gkk for gkk.save")