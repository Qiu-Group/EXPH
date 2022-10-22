from ELPH.EX_PH_scat import Gamma_scat
import numpy as np
from IO.IO_gkk import read_omega, read_gkk
from IO.IO_acv import read_Acv, read_Acv_exciton_energy
from IO.IO_common import read_bandmap, read_kmap, read_lattice,construct_kmap
from Common.common import frac2carte
from Common.progress import ProgressBar
from IO.IO_common import write_loop
import os

# input
def Exciton_Life(n_ext_acv_index=0, T=100, degaussian = 0.001, path='./',Q_kmap_start_para='nopara',Q_kmap_end_para='nopara',mute=False):
    """
    !!! Para Over Q_kmap!!!
    :param n_ext_acv_index:
    :param T:
    :param degaussian:
    :param path:
    :param Q_kmap_start_para:
    :param Q_kmap_end_para:
    :return:
    """
    # n_ext_acv_index = 0 # this is index of exciton state in Acv
    # T = 100 # randomly choosing
    # degaussian = 0.001
    outfilename = 'ex_S%s_lifetime.dat'%(n_ext_acv_index + 1)

    acv = read_Acv(path=path)

    kmap = read_kmap(path=path)  # load kmap matrix
    bandmap, occ = read_bandmap(path=path)  # load band map and number of occupation
    kmap_dic = construct_kmap(path=path)  # construct kmap dictionary {'k1 k2 k3':[0 0 0 0]}: this is used for mapping final state of scattering
    omega_mat = read_omega(path=path)  # dimension [meV]
    exciton_energy = read_Acv_exciton_energy(path=path)
    h_bar = 6.582119569E-16  # dimension = [eV.s]
    # number of point, band and phonon mode
    nc = bandmap[:, 0][-1] - occ  # load number of conduction band
    nv = occ + 1 - bandmap[:, 0][0]  # load number of valence band
    n_phonon = omega_mat.shape[1]
    Nqpt = kmap.shape[0]  # number of q-points
    bvec = read_lattice('b',path=path)
    avec = read_lattice('a',path=path)



    # progress = ProgressBar(kmap.shape[0], fmt=ProgressBar.FULL)  # progress
    if Q_kmap_start_para == 'nopara' and Q_kmap_start_para == 'nopara':
        Q_kmap_start_para = 0
        Q_kmap_end_para = kmap.shape[0]
    else:
        if type(Q_kmap_start_para) is int and type(Q_kmap_end_para) is int:
            pass
        else:
            raise Exception("the parallel parameter is not int")

    # this is good enough for parallel!!!
    # we can directly add res of every procs togather, then we can have final collected res_total
    res = np.zeros((kmap.shape[0],4))

    for Q_kmap in range(Q_kmap_start_para,Q_kmap_end_para):

        # progress.current += 1
        # progress()


        res[Q_kmap,:3] = frac2carte(bvec,kmap[Q_kmap, 0:3]) # give out bohr lattice in reciprocal space
        res[Q_kmap, 3] = 1/Gamma_scat(Q_kmap=Q_kmap, n_ext_acv_index=n_ext_acv_index, T=T, degaussian=degaussian,muteProgress=mute,path=path)

        # save temp file
        # if Q_kmap == 0:
        #     a = open('TEMP-' + outfilename, 'w')
        #     a.write(np.array2string(res[Q_kmap]).strip('[').strip(']')+'\n')
        #     a.close()
        # else:
        #     a = open('TEMP-' + outfilename, 'a')
        #     a.write(np.array2string(res[Q_kmap]).strip('[').strip(']')+'\n')
        #     a.close()
        if Q_kmap_start_para == 'nopara' and Q_kmap_start_para == 'nopara':
            # tododone: turn off this file after this loop
            write_loop(loop_index=Q_kmap,filename=outfilename,array=res[Q_kmap])

    return res
    # np.savetxt(outfilename,res)
    # os.remove('./'+'TEMP-' + outfilename)

if __name__ == "__main__":
    Exciton_Life(path='../')