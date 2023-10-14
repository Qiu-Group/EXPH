import os
import sys
import numpy as np

# required file:
# (1) .xml and .save generated from scf

mpisrun = 'srun -n 64'

biases = np.linspace(-1.7,-0.1,10)
heights = np.linspace(0.1,1,20)
fix_intensity = False
min_val, max_val = 0, 0.12


ev2Ry = 0.073498810939358

# pp_in = "&inputpp\nprefix ='VWneu'\noutdir='.',\nsample_bias=%.6f,\nplot_num=5,\n/\n&plot\niflag=3,\noutput_format=6,\nfileout='ldos.cube',\n/"

for bias in biases:
    if os.path.isdir('bias_%.3f'%bias): os.system('rm -rf bias_%.3f'%bias)
    os.mkdir('bias_%.3f'%bias)
    os.mkdir('bias_%.3f/raw_data'%bias)

    # define pp.x here
    pp_in = open("pp.in",'w')
    pp_in.write("&inputpp\n prefix ='VWneu'\n outdir='.',\n sample_bias=%.6f,\n plot_num=5,\n/\n&plot\n iflag=3,\n output_format=6,\n fileout='ldos.cube',\n/\n" % (bias * ev2Ry))
    pp_in.close()

    # run pp.x todo: modify for parallel
    os.system(mpisrun + " pp.x < pp.in")

    for h in heights:
        f_cri_in = open('cri.in','w')
        f_cri_in.write('crystal ldos.cube\nload ldos.cube\nstm height %.3f cells 1 1'% h)
        f_cri_in.close()

        # run critic 2
        os.system('critic2 cri.in')

        # replace the first three line of cri_stm.gnu
        os.system('tail -n +4 cri_stm.gnu > temp_cri_stm.gnu')
        f_cri_gnu_head = open('new_lines.txt','w')
        if fix_intensity:
            f_cri_gnu_head.write("set encoding iso_8859_1\nset terminal pdf\nset output 'stm_gnu_%.1f_%.3f.pdf'\nset cbrange [%.5f: %.5f]"%(bias, h, min_val, max_val))
        else:
            f_cri_gnu_head.write("set encoding iso_8859_1\nset terminal pdf\nset output 'stm_gnu_%.1f_%.3f.pdf'\n" % (bias, h))
        f_cri_gnu_head.close()

        os.system("cat new_lines.txt temp_cri_stm.gnu > cri_stm.gnu")
        os.system('rm temp_cri_stm.gnu')
        os.system('rm new_lines.txt')

        os.system("gnuplot cri_stm.gnu")
        os.system('mv stm_gnu_%.1f_%.3f.pdf bias_%.3f'%(bias, h, bias))
        os.system('mv cri_stm.dat bias_%.3f/raw_data/cri_stm_%.1f_%.3f.dat' % (bias,  bias, h))

