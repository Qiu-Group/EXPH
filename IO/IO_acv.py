import numpy as np
import h5py as h5
from Common.progress import ProgressBar
from Common.h5_status import print_attrs, check_h5_tree
from Common.common import move_k_back_to_BZ_1

# notice: Qpt_coor and kpt_for_each_Q are in a crystal coordinate. But kpoints in kpt_for_each_Q are in the first BZ.
# notice: eigenvalues.shape = (nQ,nS,1); eigenvectors.shape = (nQ,nS,nk,nc,nv,2)


def create_acvsh5(nQ, save_path):
    """
    :param nQ: number of Q points
    :param save_path: pth of acvs.save
    :return: create a Acv.h5
    """
    # 0.0 create 'exciton_header' and 'mf_header'
    f_temp = h5.File(save_path+'eigenvectors_2.h5','r')
    f = h5.File('Acv.h5','w')
    f_temp.copy('mf_header',f)
    f_temp.copy('exciton_header',f)
    # 1.0 'exciton_header/kpoints' modify
    f['exciton_header/kpoints/nQ'][()] = nQ
    f.create_dataset('exciton_header/kpoints/kpt_for_each_Q',data=f_temp['exciton_header/kpoints/kpts'][()])
    f.create_dataset('exciton_header/kpoints/Qpt_coor',data=np.zeros([nQ,3]))
    del f['exciton_header/kpoints/kpts']
    del f['exciton_header/kpoints/exciton_Q_shifts']
    del f['exciton_header/params/ns']
    # todo solved: exciton_Q_shifts: f.create_dataset('exciton_header/kpoints/exciton_Q_shift',data) loop add
    # 2.0 'exciton_header/params' modify
    f.create_dataset('exciton_header/params/nS', data=f_temp['exciton_header/params/nevecs'][()])
    f.create_dataset('exciton_header/params/S_total_states', data=nQ*f_temp['exciton_header/params/nevecs'][()])
    del f['exciton_header/params/nevecs']
    f_temp.close()
    # 3.0 'exciton_data'
    # i) initialization
    [nS, nc, nv, nk] = [f['exciton_header/params/nS'][()],f['exciton_header/params/nc'][()],f['exciton_header/params/nv'][()], f['exciton_header/kpoints/nk'][()]]
    f.create_dataset('exciton_data/eigenvalues', data=np.zeros(nQ*nS).reshape(nQ,nS)) # eigenvalues.shape = (nQ,nS,1)
    f.create_dataset('exciton_data/eigenvectors', data=np.zeros(nQ*nS*nk*nc*nv*2).reshape(nQ,nS,nk,nc,nv,2)) # eigenvectors.shape = (nQ,nS,nk,nc,nv,2)
    # ii) construct Acv.h5
    progress = ProgressBar(nQ, fmt=ProgressBar.FULL)
    print("creating Acv.h5")
    for i_Q in range(nQ):
        # progress:
        progress.current += 1
        progress()
        # Construct
        f_temp = h5.File(save_path+'eigenvectors_%s.h5'%(1+i_Q),'r')
        f['exciton_header/kpoints/Qpt_coor'][i_Q] = -1*f_temp['exciton_header/kpoints/exciton_Q_shifts'][()]
        f['exciton_data/eigenvalues'][i_Q] = f_temp['exciton_data/eigenvalues'][()]
        f['exciton_data/eigenvectors'][i_Q] = f_temp['exciton_data/eigenvectors'][0,:,:,:,:,0,:]
        f_temp.close()
    print("\nAcv.h5 has been created")
    f.create_dataset('mf_header/crystal/bvec_bohr', data=f['mf_header/crystal/bvec'][()] * f['mf_header/crystal/blat'][()])
    f.create_dataset('mf_header/crystal/avec_bohr', data=f['mf_header/crystal/avec'][()] * f['mf_header/crystal/alat'][()])
    del f['mf_header/crystal/bvec']
    del f['mf_header/crystal/avec']
    f.close()


def read_Acv():
    # todo: maybe try to modify it to a parallel reading
    try:
        f = h5.File("Acv.h5",'r')
    except:
        raise Exception("failed to open Acv.h5")
    acvmat = f["exciton_data/eigenvectors"][()]
    f.close()
    return acvmat

def read_Acv_exciton_energy():
    try:
        f = h5.File("Acv.h5",'r')
    except:
        raise Exception("failed to open Acv.h5")
    eigenvalue = f["exciton_data/eigenvalues"][()]
    f.close()
    return eigenvalue

# def read_Q_acv_in_crystal(path='./'):
#     f = h5.File(path+'Acv.h5','r')
#     Qpt = f['exciton_header/kpoints/Qpt_coor'][()]
#     return Qpt
#
# def read_k_acv_in_crystal(path='./'):
#     f = h5.File(path+'Acv.h5','r')
#     kpt = f['exciton_header/kpoints/kpt_for_each_Q'][()]
#     return kpt




if __name__ == "__main__":
    # todo determin save_path and nQ
    save_path = './save/acvs.save/'
    nQ = 144
    create_acvsh5(nQ,save_path)
    check_h5_tree('./Acv.h5')
    f = h5.File('Acv.h5','r')