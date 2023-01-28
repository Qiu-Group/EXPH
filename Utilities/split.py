import os


number_split = 1
number_bands = "214"
epsilon_cutoff = "30.0"

qos='development'
nodes = 40
cores = 54
time="02:00:00"
job_name="mos2"

#####################################################################################
os.system("grep '^  0.' 1-mf/2.1-wfn/kgrid.out |awk '{print $1,$2,$3}' > k_temp")
f = open("k_temp",'r')
kpoints = f.readlines()
f.close()
os.system("rm k_temp")
count = len(kpoints)
epsilon_head = "epsilon_cutoff %s\nnumber_bands %s\ncell_slab_truncation\ndont_check_norms\n\n"%(epsilon_cutoff,number_bands)
sigma_head="frequency_dependence 1\nnumber_bands %s\nband_index_min 20\nband_index_max 40\nscreening_metal\ncell_slab_truncation\ndont_check_norms\n\n"%number_bands
go_sh = ['#!/bin/bash\n', '#SBATCH --qos=%s\n'%qos, '#SBATCH --nodes=%d\n'%nodes, '#SBATCH -C knl\n', '#SBATCH -t %s\n'%time, '#SBATCH -J \n', '#SBATCH -A m3331\n', '\n', "QEPATH='/global/homes/b/bwhou/software/qe-6.7/bin'\n", "BGWPATH1='/global/homes/b/bwhou/software/BerkeleyGW-master/bin'\n", '\n', 'cd 1-epsilon\n', '#srun -n %d --cpu_bind=cores $BGWPATH1/epsilon.cplx.x > epsilon.out\n'%(nodes*cores), '\n', 'cd ../2-sigma\n', 'rm vxc.dat\n', 'rm x.dat\n', 'srun -n %d --cpu_bind=cores $BGWPATH1/sigma.cplx.x > sigma.out\n'%(nodes*cores), '\n', '\n']


for i in range(number_split):
    os.system('mkdir 2-bgw-%s'%(i+1))
    print("============================================================")
    print('part%s: 2-bgw-%s has been successfully created'%((i+1),(i+1)))
    os.chdir("2-bgw-%s"%(i+1))
    go = open('go.sh','w')
    for go_line in go_sh:
        if go_line == '#SBATCH -J \n':
            go.write('#SBATCH -J %s%s/%s \n'%(job_name, (i+1),number_split))
        else:
            go.write(go_line)
    go.close()
    os.system("chmod +x go.sh")
    os.system("mkdir 1-epsilon")
    os.system("mkdir 2-sigma")
    os.system("ln -sf ../../1-mf/2.1-wfn/WFN ../2-bgw-1/1-epsilon/WFN; ln -sf ../../1-mf/2.2-wfnq/WFN ../2-bgw-1/1-epsilon/WFNq")
    os.system("ln -sf ../../1-mf/2.1-wfn/VXC ./2-sigma/VXC; ln -sf ../../1-mf/2.1-wfn/RHO ./2-sigma/RHO;ln -sf ../../1-mf/2.1-wfn/WFN ./2-sigma/WFN_inner; ln -sf ../../2-bgw-1/1-epsilon/eps0mat.h5 ./2-sigma;ln -sf ../../2-bgw-1/1-epsilon/epsmat.h5 ./2-sigma")
    # write 1-epsilon
    if i == 0:
        os.chdir("1-epsilon")
        f_eosilon = open("epsilon.inp",'w')
        f_eosilon.write(epsilon_head)
        f_eosilon.write("begin qpoints\n")
        for j in range(count):
                if kpoints[j] == "0.000000000 0.000000000 0.000000000\n":
                    f_eosilon.write("0.001000000 0.000000000 0.000000000 1.0 1\n")
                else:
                    f_eosilon.write(" ".join(kpoints[j].split()) + " 1.0 0\n")
        f_eosilon.write("end")
        f_eosilon.close()
        print("part #%s: epsilon.inp has been created" % (i+1))
        os.chdir("../")
    # write 2-bgw
    os.chdir("2-sigma")
    f_sigma = open("sigma.inp","w")
    f_sigma.write(sigma_head)
    f_sigma.write("begin kpoints\n")
    if i+1 != number_split:
        for j in range(i*(count//number_split),i*(count//number_split)+count//number_split):
            f_sigma.write(" ".join(kpoints[j].split())+" 1.0\n")
    else:
        for j in range(i*(count//number_split),count):
            f_sigma.write(" ".join(kpoints[j].split())+" 1.0\n")
    f_sigma.write("end")
    f_sigma.close()
    print("part #%s: sigma.inp has been created" % (i+1))
    os.chdir('../../')




