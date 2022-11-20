# This script is used to calculate exciton dispersion over the 1st BZ.
# Before using this script. please make sure that 1-mf contains 4.1-wfn_co_fullgrid, where
# WFN is in a uniform k-grid like 24*24*1. By the way, you can also modify this to calculate any Q-point
# commensurate to the uniform k-grid such as: 12*12*1 Q-grid and 24*24*1 q-grid.
# Author: Bowen Hou
# Date: 11/18/2022

import os

# shift has been remove: so we don't have to remove shift in 4.1-wfn, and it may accelerate convergence speed!!

# Setting

nv = 2
nc = 1

qos='development'
nodes = 40
cores = 54
time="02:00:00"


inteqp = "number_val_bands_fine %s\nnumber_val_bands_coarse %s\n\nnumber_cond_bands_fine %s\nnumber_cond_bands_coarse %s\n\nno_symmetries_fine_grid\nno_symmetries_shifted_grid\nuse_symmetries_coarse_grid\n\nuse_momentum\n"% (nv,nv,nc,nc)
kernel = "number_val_bands %s\nnumber_cond_bands %s\n\nno_symmetries_coarse_grid\ndont_check_norms\nenergy_loss\nscreening_semiconductor\ncell_slab_truncation\n\nexciton_Q_shift 2 "%(nv, nc)
absorption = "number_val_bands_fine %s\nnumber_val_bands_coarse %s\n\nnumber_cond_bands_fine %s\nnumber_cond_bands_coarse %s\n\nno_symmetries_fine_grid\nno_symmetries_shifted_grid\nno_symmetries_coarse_grid\n\neqp_co_corrections\n\ndiagonalization\nscreening_semiconductor\ncell_slab_truncation\n\n\nuse_velocity\nenergy_resolution 0.05\ngaussian_broadening\n\nwrite_eigenvectors 10\n\nexciton_Q_shift 2 " % (nv,nv,nc,nc)

#a = ('cd 1-epsilon\n', 'srun -n %d --cpu_bind=cores $BGWPATH1/epsilon.cplx.x > epsilon.out\n'%(nodes*cores)")


########################################################################################################################################################################################################################################################
# Reading Parameter
os.system("wfn_rho_vxc_info.x ../1-mf/4.1-wfn_co_fullgrid/WFN > ./temp")
f = open("temp",'r')

while True:
    line = f.readline()
#    print(line)
    if "Number of k-points:" in line:
        n_kpt = int(line.split()[-1])
    if "   Index         Coordinates" in line:
        break

k_list = []
for i in range(n_kpt):
    line = f.readline()
    temp = line.split()[1] + ' ' + line.split()[2] + ' ' + line.split()[3] 
    k_list.append(temp)

f.close()

# shift has been remove: so we don't have to remove shift in 4.1-wfn, and it may accelerate convergence speed!!
shift = k_list[0]

shift = [float(x) for x in shift.split()]
k_list_new = []
for i in range(len(k_list)):
    temp_1 = k_list[i].split()
    temp_2 = [float(x) for x in temp_1]
    for j in range(len(temp_2)):
        temp_2[j] = temp_2[j] - shift[j]

    k_list_new.append("%.6f %.6f %.6f"%(-1*temp_2[0], -1*temp_2[1], -1*temp_2[2]))
k_list = k_list_new    

#os.system("rm temp")
os.system("mkdir 5_Q")
os.system("mv temp ./5_Q/kpt.dat")
os.system("mkdir 5_Q/inteqp")
f = open("inteqp.inp",'w')
f.write(inteqp)
f.close()

os.system("mv inteqp.inp 5_Q/inteqp/")
os.chdir("./5_Q/inteqp")
os.system("ln -sf ../../../1-mf/2.3-wfn_co/WFN ./WFN_co")
os.system("ln -sf ../../../1-mf/4.1-wfn_co_fullgrid/WFN ./WFN_fi")
os.system("ln -sf ../../2.1-sigma/eqp1.dat ./eqp_co.dat")

os.chdir("../")
for i in range(len(k_list)):
    print(i,'/',len(k_list))

    os.system("mkdir Q-%s"%(i+1))
    os.system("mkdir Q-%s/5.1-kernel-Q"%(i + 1))
    os.system("mkdir Q-%s/5.2-absorp-Q"%(i + 1))
    os.chdir("./Q-%s"% (i + 1))
    os.system("echo %s > kpt"%k_list[i])
    os.chdir("./5.1-kernel-Q")
    os.system("ln -sf ../../../1-epsilon/eps0mat.h5 ./")
    os.system("ln -sf ../../../1-epsilon/epsmat.h5 ./")
    os.system("ln -sf ../../../../1-mf/4.1-wfn_co_fullgrid/WFN ./WFN_co")
    kernel_temp = kernel + k_list[i]
    f = open("kernel.inp",'w')
    f.write(kernel_temp)
    f.close()
    os.chdir("../5.2-absorp-Q")
    os.system("ln -sf ../5.1-kernel-Q/bsemat.h5 ./")
    os.system("ln -sf ../../../1-epsilon/eps0mat.h5 ./")
    os.system("ln -sf ../../../1-epsilon/epsmat.h5 ./")
    os.system("ln -sf ../../../../1-mf/4.1-wfn_co_fullgrid/WFN ./WFN_co")
    os.system("ln -sf ../../../../1-mf/4.1-wfn_co_fullgrid/WFN ./WFN_fi")
    os.system("ln -sf ../../../../1-mf/4.1-wfn_co_fullgrid/WFN ./WFNq_fi")
    os.system("ln -sf ../../inteqp/eqp.dat ./eqp_co.dat")
    absorption_temp = absorption + k_list[i]
    f = open("absorption.inp",'w')
    f.write(absorption_temp)
    f.close()
    
    os.chdir("../../")


go_sh_tacc = ['#!/bin/bash\n','#SBATCH -J fin_Q\n','#SBATCH -o myjob.o\%j\n' '#SBATCH -e myjob.e\%j\n',

              '#SBATCH -p %s\n'%qos, '#SBATCH -N %d\n'%nodes, '#SBATCH -n %d\n'%(nodes*cores), '#SBATCH -t %s\n'%time,  '#SBATCH -A PHY20032\n', '\n',




         "#QEPATH='/global/homes/b/bwhou/software/qe-6.7/bin'\n",



         "BGWPATH1='/home1/08237/bwhou/software/BerkeleyGW-3.0.1/bin/'\n", '\n\n',
         "cd inteqp\nibrun $BGWPATH1/inteqp.cplx.x > inteqp.out\n",
         "cd ../\n",
         "for ((i=2;i<=%s;i++));\ndo\n"%n_kpt,
         "cd ./Q-$i/5.0-inteqp-Q\nibrun   $BGWPATH1/inteqp.cplx.x > inteqp.out\n",
         "cd ../5.1-kernel-Q\nibrun   $BGWPATH1/kernel.cplx.x > kernel.out\n",
         "cd ../5.2-absorp-Q\n","ibrun  $BGWPATH1/absorption.cplx.x > absorption.out\n","cd ../../\ndone"]



go = open('go.sh','w')
for go_line in go_sh_tacc:
    go.write(go_line)
go.close()

os.system("chmod +x go.sh")



# mkdir for each kpoint

