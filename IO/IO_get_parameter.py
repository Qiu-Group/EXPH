import os
import h5py as h5

# save_path = './save_MoS2_symm/'

# (1) get nQ
# nQ = len(os.listdir(save_path + 'acvs.save/')) - 1

# (2) get nk


def read_kkqQ_number(save_path):
    """
    :param save_path: ./save/
    :return:
    """
    nQ = Q_number_acvs(save_path)
    nk_acv = k_number_acvs(save_path)
    nq = q_number_gkk(save_path)
    nk_gkk = k_number_gkk(save_path)
    nmode = mode_number(save_path)
    print("nQ = %s, nk_acv =%s, nk_gkk = %s, nq = %s, nmode = %s" % (nQ,nk_gkk,nk_gkk,nq,nmode))
    return nQ, nq, min(nk_acv, nk_gkk), nmode


def Q_number_acvs(save_path):
    """
    :param save_path:  save/
    :return:
    """
    if os.path.isfile(save_path+"/acvs.save/kgrid.log"):
        nQ = len(os.listdir(save_path + 'acvs.save/')) - 1
    else:
        nQ = len(os.listdir(save_path + 'acvs.save/'))
    return nQ

def k_number_acvs(save_path):
    """
    :param save_path:  save/
    :return:
    """
    f = h5.File(save_path+'/acvs.save/eigenvectors_2.h5','r')
    nk= f['exciton_header/kpoints/nk'][()]
    f.close()
    return nk

def q_number_gkk(save_path):
    return counter_lines(save_path+'/gkk.save/q.dat')

def k_number_gkk(save_path):
    return counter_lines(save_path + '/gkk.save/k.dat')

def counter_lines(file_path):
    a = open(file_path,'r')
    lenth = len(a.readlines())
    a.close()
    return lenth

def mode_number(save_path):
    f = h5.File(save_path+'/acvs.save/eigenvectors_2.h5','r')
    natom= f['mf_header/crystal/nat'][()]
    f.close()
    return natom * 3
