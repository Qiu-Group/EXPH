# This only works for Linux
# Nothing to set, just make sure neccessary files:
# in_head/pp_in (2.1-wfn), inq_head/pp_inq(2.2-wfnq), kgrid.inp, kpt.dat(get from 4-path/WFN)
# kpt.dat could store any k-points(Q-points) you wan to calculate.
import numpy as np
import os

###########-- Setting --##########
qos='development'
nodes = 40
cores = 54
time="02:00:00"
##################################

def kgrid_inp_list(kgrid_lines,Q_shift):
    """
    :param kgrid_lines: [['24 24 1\n', '0.000000000000 0.000000000000 0.000000000000\n', '0.050000000000000003  -0.0  0.0\n', ...]
    :param Q_shift: array([ 0.166667,  0.      ,  0.      ])
    :return:[['24 24 1\n', '0.000000000000 0.000000000000 0.000000000000\n', 'Q_shift', ...]  compared with kgrid_lines
    """
    # Q_shift_temp = Q_shift_list[3] #i
    # f = open('kgrid.inp','r')
    kgrid_inp_temp= []
    for i in range(len(kgrid_lines)):
        if i ==2:
            kgrid_inp_temp.append("%.7f %.7f %.7f"%(Q_shift[0],Q_shift[1],Q_shift[2])+'\n')
        else:
            kgrid_inp_temp.append(kgrid_lines[i])
    return kgrid_inp_temp

def pp_inq_list(pp_inq_lines,Q_shift):
    for j,line in enumerate(pp_inq_lines):
        if 'wfng_nk1' in line:
            wfng_nk1 = int(line.split()[-1])
        if 'wfng_dk1' in line:
            # print('wfng_nk1',wfng_nk1)
            # print('Q_shift[0]', Q_shift[0])
            pp_inq_lines[j] = '   wfng_dk1 = %.4f \n' % (wfng_nk1 * Q_shift[0])
        if 'wfng_dk2' in line:
            pp_inq_lines[j] = '   wfng_dk2 = %.4f \n' % (wfng_nk1 * Q_shift[1])
        if 'wfng_dk3' in line:
            pp_inq_lines[j] = '   wfng_dk3 = %.4f \n' % (wfng_nk1 * Q_shift[2])
    return pp_inq_lines

def pp_inq_list_2(pp_inq_lines):
    for j,line in enumerate(pp_inq_lines):
        # if 'wfng_nk' in line or 'wfng_dk' in line:
        if 'wfng_dk' in line:
            pp_inq_lines[j] = ''
    return pp_inq_lines

def write_list_line_by_line(a_list, file_name):
    f = open(file_name,'w')
    for i in range(len(a_list)):
        f.write(a_list[i])
    f.close()


def get_unfold_grid_from_klog():
    """
    :return: list to be passed to wrte_list_line_by_line.
    Read unfolded kgrid from kgrid.log
    """
    f = open('kgrid.log','r')
    lines = f.readlines()
    nk_index = "None"
    start_index = "None"
    for i, line in enumerate(lines):
        if "k-points in the unfolded Brillouin Zone" in line:
            start_index = i
            nk_index = i+1
            break
    output_list = ["K_POINTS crystal\n", lines[nk_index]]
    for i in range(int(lines[nk_index])):
        j = i + start_index + 2
        output_list.append(' '.join(lines[j].split()[1:])+'\n')
    return output_list

def get_unfold_grid_from_klog_without_shift():
    """
    :return: list to be passed to wrte_list_line_by_line.
    Read unfolded kgrid from kgrid.log
    """
    f = open('kgrid.log','r')
    lines = f.readlines()
    nk_index = "None"
    start_index = "None"
    for i, line in enumerate(lines):
        if "k-points in the original uniform grid" in line:
            start_index = i
            nk_index = i+1
            break
    output_list = ["K_POINTS crystal\n", lines[nk_index]]
    for i in range(int(lines[nk_index])):
        j = i + start_index + 2
        output_list.append(' '.join(lines[j].split()[1:4])+' 1.0\n')
    return output_list

if __name__ == "__main__":
    # write_list_line_by_line(kgrid_inp_list(kgrid_lines,Q_shift_list[2]), 'test.inp')
    pass

    # (0) Checking and Loading
    # (i) check in_head/pp_in (2.1-wfn), inq_head/pp_inq(2.2-wfnq), kgrid.inp, kpt.dat(get from 4-path/WFN)
    if os.path.isfile('./in_head') and os.path.isfile('./inq_head') and os.path.isfile('pp_in') and os.path.isfile('pp_inq') and os.path.isfile(
            './kgrid.inp') and os.path.isfile('./kpt.dat'):
        print(('in_head/pp_in, inq_head/pp_inq, pp_in and pp_inq found'))
    else:
        raise Exception("in_head/pp_in, inq_head/pp_inq, pp_in and pp_inq is missing")
    Q_shift_list = np.loadtxt('kpt.dat')
    n_Q_shift = Q_shift_list.shape[0]
    kgrid_inp = open('./kgrid.inp', 'r')
    kgrid_lines = kgrid_inp.readlines()


    scf = os.listdir('./1-scf')
    save_file = "none"
    for i_file in scf:
        if 'save' in i_file:
            save_file = i_file


    # (1) build 4.1-wfn_co_fullgrid (w/o any shfit)
    print('building 4.1-wfn_co_fullgrid')
    os.system('mkdir ./4.1-wfn_co_fullgrid')
    # print(os.getcwd())
    os.chdir('./4.1-wfn_co_fullgrid')
    # print(os.getcwd())
    write_list_line_by_line(kgrid_inp_list(kgrid_lines=kgrid_lines, Q_shift=-1*np.array([0,0,0])), file_name='./kgrid.inp')
    os.system('cp ../in_head ./in')
    os.system('kgrid.x kgrid.inp kgrid.out kgrid.log')
    # write_list_line_by_line(get_unfold_grid_from_klog() ,'kgrid_unfold.out')
    os.system('cat kgrid.out >> in')
    os.system('cp -r ../1-scf/pp ./')
    os.system('cp ../pp_in ./')
    os.system('mkdir '+ save_file)
    os.chdir(save_file)
    os.system('ln -sf ../../1-scf/'+save_file+'/data-file-schema.xml ./')
    os.system('ln -sf ../../1-scf/'+save_file+'/charge-density.dat ./')
    os.chdir('../../')

    # print(Q_shift_list)
    # (2) write a loop to build 4.2-wfnq_co-n (w/ Q shift)
    for i in range(n_Q_shift):
        print('building 4.2-wfnq_co-%s out of %s'% (i+1,n_Q_shift))
        os.system('mkdir 4.2-wfnq_co-%s'%(i+1))
        os.chdir('4.2-wfnq_co-%s'%(i+1))
        write_list_line_by_line(kgrid_inp_list(kgrid_lines=kgrid_lines,Q_shift=-1*Q_shift_list[i]), file_name='./kgrid.inp')
        os.system('cp ../inq_head ./in')
        os.system('kgrid.x kgrid.inp kgrid.out kgrid.log')

        if np.all(Q_shift_list[i] == 0.):
            # print("np.all(Q_shift_list[%s]) == 0"%i)
            write_list_line_by_line(get_unfold_grid_from_klog_without_shift(), 'kgrid_unfold.out')
            # os.system('cat kgrid.out >> in')
        else:
            write_list_line_by_line(get_unfold_grid_from_klog(), 'kgrid_unfold.out')
        os.system('cat kgrid_unfold.out >> in')

        os.system('cp -r ../1-scf/pp ./')
        os.system('cp ../pp_inq ./pp_inq') ##tododone: repace some part of pp_inq
        # read pp
        pp_inq = open('pp_inq', 'r')
        pp_inq_lines = pp_inq.readlines()
        write_list_line_by_line(pp_inq_list_2(pp_inq_lines=pp_inq_lines),'pp_in') # this is for shift of dkg 1,2,3
        # write_list_line_by_line(pp_inq_lines, 'pp_in')

        os.system('mkdir ' + save_file)
        os.chdir(save_file)
        os.system('ln -sf ../../1-scf/' + save_file + '/data-file-schema.xml ./')
        os.system('ln -sf ../../1-scf/' + save_file + '/charge-density.dat ./')
        os.chdir('../../')

    go_sh_tacc = ['#!/bin/bash\n', '#SBATCH -J fin_Q\n', '#SBATCH -o myjob.o\%j\n' '#SBATCH -e myjob.e\%j\n',

                  '#SBATCH -p %s\n' % qos, '#SBATCH -N %d\n' % nodes, '#SBATCH -n %d\n' % (nodes * cores),
                  '#SBATCH -t %s\n' % time, '#SBATCH -A PHY20032\n', '\n',

                  "QEPATH='/home1/08237/bwhou/software/qe-6.7/bin'\n",
                  "BGWPATH1='/home1/08237/bwhou/software/BerkeleyGW-3.0.1/bin/'\n", '\n\n',
                  "cd 4.1-wfn_co_fullgrid\nibrun $QEPATH/pw.x -nk 16 -input in > in.out\nibrun $QEPATH/pw2bgw.x -nk 16 -input pp_in > pp_out\n",
                  "cd ../\n",
                  "for ((i=1;i<=%s;i++));\ndo\n" % n_Q_shift,
                  "cd ./4.2-wfnq_co-$i\nibrun $QEPATH/pw.x -nk 16 -input in > in.out\nibrun $QEPATH/pw2bgw.x -nk 16 -input pp_in > pp_out\n",
                  "cd ../\ndone"]
    write_list_line_by_line(go_sh_tacc,'go.sh')
    os.system('chmod +x go.sh')