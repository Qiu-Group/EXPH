import numpy as np
import h5py as h5
from Common.progress import ProgressBar
from Common.h5_status import print_attrs, check_h5_tree
from Common.common import move_k_back_to_BZ_1
import os

def readkkqQ(readwhat=0, h5path="./"):
    """
    read qpt, kpt, kpt or Qpt from h5 file
    :param readwhat: <1>:Qpt  <2>:kpt_acv  <3>:qpt  <4>:kpt_gkk
    :param h5path:
    :return: k-grid in 1st BZ
    """
    acv = h5.File(h5path + 'Acv.h5', 'r')
    gkk = h5.File(h5path + 'gkk.h5', 'r')
    if readwhat == 0:
        print("<1>:Qpt  <2>:kpt_acv  <3>:qpt  <4>:kpt_gkk")
    elif readwhat == 1:
        return move_k_back_to_BZ_1(acv['exciton_header/kpoints/Qpt_coor'][()])
    elif readwhat == 2:
        return move_k_back_to_BZ_1(acv['exciton_header/kpoints/kpt_for_each_Q'][()])
    elif readwhat == 3:
        return move_k_back_to_BZ_1(gkk['epw_header/qpt_coor'][()])
    elif readwhat == 4:
        return move_k_back_to_BZ_1(gkk['epw_header/kpt_coor'][()])
    else:
        raise Exception("input should be: 0-5")

def read_kmap():
    try:
        kmap = np.loadtxt("kkqQmap.dat")
    except:
        raise Exception("failed to open kmap.dat")
    return kmap

def read_bandmap():
    try:
        bandmap = np.loadtxt("bandmap.dat")
    except:
        raise Exception("failed to open bandmap.dat")
    f = open("bandmap.dat",'r')
    a = f.readline()
    f.close()
    occ = int(a.split()[-1])
    return [bandmap, occ]

def write_loop(loop_index,filename,array):
    if loop_index == 0:
        a = open(filename, 'w')
        a.write(np.array2string(array).strip('[').strip(']') + '\n')
        a.close()
    else:
        a = open(filename, 'a')
        a.write(np.array2string(array).strip('[').strip(']') + '\n')
        a.close()

def read_lattice(lattice_type='a'):
    """
    :param lattice_type: a -> real lattice; b -> reciprocal lattice
    :return:
    """
    try:
        acv = h5.File('Acv.h5','r')
    except:
        raise Exception("failed to open Acv.h5")
    if lattice_type == 'a':
        return acv['mf_header/crystal/avec_bohr'][()]
    else:
        return acv['mf_header/crystal/bvec_bohr'][()]

def construct_kmap():
    #to be filled
    try:
        kmap = np.loadtxt('kkqQmap.dat')
    except:
        raise Exception("failed to open kkqQmap.dat")
    res = {}
    for i in range(kmap.shape[0]):
        res['  %.5f    %.5f    %.5f' % (kmap[i, 0:3][0], kmap[i, 0:3][1], kmap[i, 0:3][2])] = kmap[i, 3:]
    return res

if __name__ == "__main__":
    pass
