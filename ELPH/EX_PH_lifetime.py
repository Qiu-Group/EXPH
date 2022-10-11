from ELPH.EX_Ph_scat import Gamma_scat
import numpy as np
from IO.IO_gkk import read_omega, read_gkk
from IO.IO_acv import read_Acv, read_Acv_exciton_energy
from IO.IO_common import read_bandmap, read_kmap
from Common.kgrid_check import construct_kmap


# input
def Exciton_Life(n_ext_acv_index=0, T=100, degaussian = 0.001):
    # n_ext_acv_index = 0 # this is index of exciton state in Acv
    # T = 100 # randomly choosing
    # degaussian = 0.001
    outfilename = 'ex_S%s_lifetime.dat'%(n_ext_acv_index + 1)

    kmap = read_kmap()  # load kmap matrix
    bandmap, occ = read_bandmap()  # load band map and number of occupation
    kmap_dic = construct_kmap()  # construct kmap dictionary {'k1 k2 k3':[0 0 0 0]}: this is used for mapping final state of scattering
    omega_mat = read_omega()  # dimension [meV]
    exciton_energy = read_Acv_exciton_energy()
    h_bar = 6.582119569E-16  # dimension = [eV.s]
    # number of point, band and phonon mode
    nc = bandmap[:, 0][-1] - occ  # load number of conduction band
    nv = occ + 1 - bandmap[:, 0][0]  # load number of valence band
    n_phonon = omega_mat.shape[1]
    Nqpt = kmap.shape[0]  # number of q-points


    res = np.zeros((kmap.shape[0],4))
    for Q_kmap in range(kmap.shape[0]):
        res[Q_kmap,:3] = kmap[Q_kmap, 0:3]
        res[Q_kmap, 3] = 1/Gamma_scat(Q_kmap=Q_kmap, n_ext_acv_index=n_ext_acv_index, T=T, degaussian=degaussian)

    np.savetxt(outfilename,res)


if __name__ == "__main__":
    Exciton_Life()