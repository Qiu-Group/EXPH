import struct
import numpy as np
import os
import h5py as h5
from Common.progress import ProgressBar
from Common.h5_status import check_h5_tree

"""
input: gkk.save, nk, nq, nmode, ni, nj
outout: gkk.h5
"""
# Note: gkk.shape = (nq,nk,ni,nj,nmode)

# Note:C
# In order to get elphmat.dat, run following in command line (all data are from epw.out):
# todo: get above information of nq, nk, nmode, i, j from save: write a script to get file and information
#=============================================================================================
# cat epw.out | grep -E '^ +[0-9]+ +[0-9]+ +[0-9]+'|awk '{print $7}' > elphmat.dat
# grep '     iq' epw.out |awk '{print $5,$6,$7}' > q.dat
# grep 'lattice parameter ' epw.out |awk '{print $5}' >a0.dat
# grep '            a' epw.out|awk '{print $4,$5,$6}' > a.dat
# grep '            b' epw.out|awk '{print $4,$5,$6}' > b.dat
# kmesh.pl 12 12 1 |grep '^ '|awk '{print $1,$2,$3}' > k.dat #todo: 12 12 1 should be determined by smarter method..
# grep 'ik =       1' epw.out -A 11| grep -E '^ +[0-9]+ +[0-9]+ +[0-9]+'|awk '{print $6}' > omega.dat #todo: notice that 11 is determined by nmode + 2
#=============================================================================================
# then put all data to save/gkk.save



def create_gkkh5(nq,nk,nmode,ni,nj,save_path):
    """
    :param nq: number of q points
    :param nk: number of k points
    :param nmode: number of phonon mode
    :param ni: number of initial electron states
    :param nj: number of final electron states
    :param save_path: gkk.save
    :return: create gkk.h5
    """
    print("creating gkk.h5\n")
    # 1.0 load raw data (raw means the data need to be reshape)
    data_temp_raw = np.loadtxt(save_path+'elphmat.dat')
    omega_raw = np.loadtxt(save_path+"omega.dat")
    k_temp = np.loadtxt(save_path+'k.dat')
    q_temp = np.loadtxt(save_path+'q.dat') #
    a_temp = np.loadtxt(save_path+'a.dat')
    b_temp = np.loadtxt(save_path+'b.dat')
    a0_temp = np.loadtxt(save_path+'a0.dat')

    # 1.1 check data
    if len(data_temp_raw) != nq*nk*nmode*ni*nj:
        raise Exception("nq*nk*nmode*ni*nj != gkk.shape")
        # todo: add a break in the final version of function
    if q_temp.shape[0] != nq:
        raise Exception("nq != q_temp.shape")
    if q_temp.shape[0] != nq:
        raise Exception("nk != k_temp.shape")
    if omega_raw.shape[0] != nq*nmode:
        raise Exception("nq*nmode != omega_temp.shape")

    omega_temp = omega_raw.reshape(nq,nmode)
    data_temp = data_temp_raw.reshape(nq,nk,ni,nj,nmode)
    # 2.0 create gkk.h5 file
    f = h5.File('gkk.h5','w')
    header = f.create_group('epw_header')
    f.create_dataset('epw_data/elphmat',data=data_temp)
    header.create_dataset('omega',data=omega_temp)
    header.create_dataset('kpt_coor',data=k_temp)
    header.create_dataset('qpt_coor',data=q_temp)
    header.create_dataset('a0_bohr',data=a0_temp)
    header.create_dataset('a_lattice_bohr',data=a_temp*a0_temp) # in bohr
    header.create_dataset('b_lattice_bohr',data=b_temp*(2*np.pi/a0_temp)) # in 2pi/bohr
    print("\nAcv.h5 has been created")
    f.close()


def read_gkk(path="./"):
    try:
        f=h5.File(path+'gkk.h5','r')
    except:
        raise Exception("failed to open gkk.h5")
    gkkmat = f['epw_data/elphmat'][()]
    f.close()
    return gkkmat

def read_omega(path='./'):
    try:
        f = h5.File(path+'gkk.h5','r')
    except:
        raise Exception("failed to open gkk.h5")
    omega_temp = f['epw_header/omega'][()]
    return omega_temp

if __name__ == "__main__":
    nq = 16
    nk = 144
    nmode = 9
    ni = 4
    nj = 4
    save_path = './save/gkk.save/'

    create_gkkh5(nq,nk,nmode,ni,nj,save_path)
    check_h5_tree('gkk.h5')
    f = h5.File('gkk.h5','r')