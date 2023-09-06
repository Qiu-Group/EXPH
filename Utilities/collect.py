#!/usr/bin/env python

import os
import sys
from Utilities.epmatq import read_epb, read_omega
from Utilities.rotate import epbfile

# TODO!!
#for i in {1..3};do cp Q-$i/5.2-absorp-Q/eigenvectors.h5 save/eigenvectors_$i.h5;done
if len(sys.argv) != 3:
    print("Usage[acvs]: acv nQ")
    # print("Usage[gkk]: gkk n_lambda")
    print("Usage[gkk]: gkk prefix")
    print("nQ: number of n-Q\nprefix: name of system")
    sys.exit(1)

second_parameter = sys.argv[2]

if sys.argv[1] == 'acv':
    os.mkdir('./acvs.save')
    os.system('for i in {1..%s};do cp Q-$i/5.2-absorp-Q/eigenvectors.h5 acvs.save/eigenvectors_$i.h5;done'%second_parameter)
    print('data collected!')

elif sys.argv[1] == 'gkk':
    # read cart electron-phonon matrix
    read_epb(prefix=second_parameter)
    # rotate el_ph_cart to el_ph_phonon
    epb = epbfile(prefix="MoS2")
    epb.diagonalize_dynmat()
    epb.write_hdf5_qbyq()

    os.mkdir('./gkk.save')
    os.mkdir('./gkk.save/backup_parameter')
    os.system('cp epw.out gkk.save')

    os.system('mv elphmat.h5 gkk.save/')
    os.system('mv elphmat_phase.h5 gkk.save/')
    os.system('mv omega.dat gkk.save')
    os.system('mv k.dat gkk.save')
    os.system('mv q.dat gkk.save')
    os.system('mv a.dat gkk.save')
    os.system('mv a0.dat gkk.save')
    os.system('mv b.dat gkk.save')

    os.system('mv q_car_elph.dat gkk.save/backup_parameter')
    os.system('mv q_frac_omega.dat gkk.save/backup_parameter')
    os.system('mv k_car_elph.dat gkk.save/backup_parameter')
    os.system('mv qelph_qomega_qmap.dat gkk.save/backup_parameter')
    os.system('mv omega_uniform_q.dat gkk.save/backup_parameter')
    # nmode, _, _, _, _, _, _ = read_parameter()
    os.chdir('./gkk.save')

    # os.system("grep 'lattice parameter ' epw.out |awk '{print $5}' >a0.dat")
    # os.system("grep '            a' epw.out|awk '{print $4,$5,$6}' > a.dat")
    # os.system("grep '            b' epw.out|awk '{print $4,$5,$6}' > b.dat")

    # os.system("grep 'q(' epw.out |awk '{print $6,$7,$8}' > q_car_elph.dat")
    # os.system("grep '     iq' epw.out |awk '{print $5,$6,$7}' > q_frac_omega.dat")
    # os.system("grep 'k(' epw.out |awk '{print $5,$6,$7}'|tr -d ')'|tr -d ',' > k_car_elph.dat")

    # TODO_done (1): find a way to transfer elphmat to [meV]
    # TODO_done (2): find a way to calculate k and q points in fractional coordinate.



#----------------------------------------
    # os.system("cat epw.out | grep -E '^ +[0-9]+ +[0-9]+ +[0-9]+'|awk '{print $7}' > elphmat.dat")
    # os.system("cat epw.out | grep -E '^ +[0-9]+ +[0-9]+ +[0-9]+'|awk '{print $8,$9}' > elphmat_phase.dat")
    # os.system("grep '     iq' epw.out |awk '{print $5,$6,$7}' > q.dat")
    # os.system("grep 'lattice parameter ' epw.out |awk '{print $5}' >a0.dat")
    # os.system("grep '            a' epw.out|awk '{print $4,$5,$6}' > a.dat")
    # os.system("grep '            b' epw.out|awk '{print $4,$5,$6}' > b.dat")
    # os.system("head epw.out -n 5000|grep 'Using uniform q-mesh:' |awk '{print $4,$5,$6}' > mesh_temp")

    # temp = open('mesh_temp','r')
    # mesh = temp.readline().strip()
    # temp.close()
    # os.system('rm mesh_temp')

    # os.system("kmesh.pl %s |grep '^ '|awk '{print $1,$2,$3}' > k.dat"%mesh)
    # os.system("grep 'ik =       1' epw.out -A %s| grep -E '^ +[0-9]+ +[0-9]+ +[0-9]+'|awk '{print $6}' > omega.dat"%(int(second_parameter) + 2))
    # os.system("grep 'ik =       1' epw.out -A %s| grep -E '^ +[0-9]+ +[0-9]+ +[0-9]+'|awk '{print $6}' > omega.dat" % (nmode + 2))
    print('data collected!')

else:
    raise Exception("the first input can only be acv or gkk; acv for acvs.save and gkk for gkk.save")