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

def read_kmap(path="./"):
    try:
        kmap = np.loadtxt(path+'kkqQmap.dat')
    except:
        raise Exception("failed to open kmap.dat")
    return kmap

def read_bandmap(path="./"):
    try:
        bandmap = np.loadtxt(path+'bandmap.dat')
    except:
        raise Exception("failed to open bandmap.dat")
    f = open(path+'bandmap.dat','r')
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

def read_lattice(lattice_type='a',path='./'):
    """
    :param lattice_type: a -> real lattice; b -> reciprocal lattice
    :return:
    """
    try:
        acv = h5.File(path+'Acv.h5','r')
    except:
        raise Exception("failed to open Acv.h5")
    if lattice_type == 'a':
        return acv['mf_header/crystal/avec_bohr'][()]
    else:
        return acv['mf_header/crystal/bvec_bohr'][()]

def construct_kmap(path='./'):
    #to be filled
    try:
        kmap = np.loadtxt(path+'kkqQmap.dat')
    except:
        raise Exception("failed to open kkqQmap.dat")
    res = {}
    for i in range(kmap.shape[0]):
        res['  %.5f    %.5f    %.5f' % (kmap[i, 0:3][0], kmap[i, 0:3][1], kmap[i, 0:3][2])] = kmap[i, 3:]
    return res

def construct_symmetry_dic(save_path,mute=False):
    """
    :param save_path: the path of acvs.save/
    :param uniform_grid_array: see read_kgrid_log
    :param reduced_grid_array: see read_kgrid_log
    :return: symm_dic: {'  %.6f    %.6f    %.6f'}
    """
    uniform_grid_array, reduced_grid_array = read_kgrid_log(save_path, mute=mute)
    n_uniform = uniform_grid_array.shape[0]
    n_reduced = reduced_grid_array.shape[0]

    full2reduce_dic = {}
    for j in range(n_reduced):
        full2reduce_dic[reduced_grid_array[j,4]] = reduced_grid_array[j,3]

    symm_dic = {}
    for i in range(n_uniform):
        if int(uniform_grid_array[i,4]) == 0:
            symm_dic['  %.6f    %.6f    %.6f' % (uniform_grid_array[i][0],uniform_grid_array[i][1], uniform_grid_array[i][2])] = int(full2reduce_dic[uniform_grid_array[i][3]]) - 1
        else:
            symm_dic['  %.6f    %.6f    %.6f' % (uniform_grid_array[i][0], uniform_grid_array[i][1], uniform_grid_array[i][2])] = int(full2reduce_dic[uniform_grid_array[i][4]]) - 1

    return symm_dic

def read_kgrid_log(save_path, mute=False):
    """
    :param save_path: the path of acvs.save/
    :return: uniform_grid_array --> np.array([kx,ky,kz,k_index_uniform,k_index_uniform_after_reduced])
             reduced_grid_array --> np.array([kx,ky,kz ,k_index_reduced ,k_index_uniform])
    """
    kgrid_file = open(save_path + "/kgrid.log",'r')

    while True:
        line = kgrid_file.readline()
        if "k-points in the original uniform grid" in line:
            uniform_kgrid_number = int(kgrid_file.readline())
            uniform_grid_array = np.zeros((uniform_kgrid_number,5))
            if not mute:
                print("%s uniform kgrid found in kgrid.log"%uniform_kgrid_number)
            for i in range(uniform_kgrid_number):
                line = kgrid_file.readline()
                [k_index_uniform, kx, ky, kz, weight, k_index_uniform_after_reduced] = list(map(float,line.split()[0:6]))
                uniform_grid_array[i] = np.array([kx,ky,kz,k_index_uniform,k_index_uniform_after_reduced])

        if "k-points in the irreducible wedge" in line:
            reduced_kgrid_number = int(kgrid_file.readline())
            reduced_grid_array = np.zeros((reduced_kgrid_number,5))
            if not mute:
                print("%s uniform kgrid found in kgrid.log"%reduced_kgrid_number)
            for i in range(reduced_kgrid_number):
                line = kgrid_file.readline()
                [k_index_reduced, kx, ky, kz, weight, k_index_uniform] = list(map(float,line.split()[0:6]))
                reduced_grid_array[i] = np.array([kx,ky,kz ,k_index_reduced ,k_index_uniform])
            break
    return uniform_grid_array, reduced_grid_array


if __name__ == "__main__":
    # a,b = read_kgrid_log('../save_hBN_symm/')
    sym_dic = construct_symmetry_dic('../save_hBN_symm/acvs.save')
    pass
