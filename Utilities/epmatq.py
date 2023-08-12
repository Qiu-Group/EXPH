import numpy as np
from scipy.io import FortranFile
import sys
import os
from Common.common import equivalence_order, move_k_back_to_BZ_1
import h5py as h5

ry2meV = 13605.662285137
# This script is used to read epmatq from .epb file and reorder electron-phonon matrix based on Fortran order
# This should be integrated into collect.py
# TODO_done: (1) See how does k pool (uniform or non-uniform) affect parallel of epb
# TODO_done: (2) Find a way to collect ni nj nk nq and nmu
# TODO_done: (3) See how excludes bands affect ni and nj


# TODO_done: check unit of epmatq, transfer it to [meV]
# TODO_done: try to read this from different pool and joint them together


def read_epb(prefix): # elphmat*.dat is generated here
    """
    :param prefix:
    :return: el_ph_phase and el_ph_nophase, but they are ordered by q points, which are firstly looped over q in irr wedge then star.
    """
    nmu, nbnd_calculated, nbnd_total, nbnd_exclude, nk, nq, number_of_epb = read_parameter()
    ni, nj = nbnd_calculated, nbnd_calculated
    if nk % number_of_epb != 0: raise Exception('nk % number_of_epb != 0')
    nk_each_pool = nk // number_of_epb
    print('number of k in each pool:',nk_each_pool)

    # TODO_done: find a better way to concatenate matrix together. (build a res matrix before)

    # final_ep_real_epb1, final_ep_imag_epb1, final_g2_epb1 = read_single_epb(prefix,1,ni,nj,nk_each_pool,nq,nmu)

    final_ep_real, final_ep_imag, final_g2 = np.zeros((nk,number_of_epb * nk_each_pool,ni,nj,nmu)), np.zeros((nk,number_of_epb * nk_each_pool,ni,nj,nmu)), np.zeros((nk,number_of_epb * nk_each_pool,ni,nj,nmu))
    # final_ep_real[:, :nk_each_pool, :, :, :] = final_ep_real_epb1
    # final_ep_imag[:, :nk_each_pool, :, :, :] = final_ep_imag_epb1
    # final_g2[:, :nk_each_pool, :, :, :] = final_g2_epb1

    print('shape of final_ep_real, final_ep_imag, final_g2',final_ep_real.shape, final_ep_imag.shape, final_g2.shape)
    for iepb in range(1,number_of_epb+1):
        temp_ep_real, temp_ep_imag, temp_g2 = read_single_epb(prefix, iepb)
        # final_ep_real = np.concatenate((final_ep_real,temp_ep_real),axis=1)
        # final_ep_imag = np.concatenate((final_ep_imag, temp_ep_imag), axis=1)
        # final_g2 = np.concatenate((final_g2, temp_g2), axis=1)
        final_ep_real[:,(iepb-1)*nk_each_pool:iepb*nk_each_pool,:,:,:] = temp_ep_real
        final_ep_imag[:,(iepb-1)*nk_each_pool:iepb*nk_each_pool,:,:,:] = temp_ep_imag
        final_g2[:,(iepb-1)*nk_each_pool:iepb*nk_each_pool,:,:,:] = temp_g2


    # This step will transfer gkk to [meV]
    # inv_sqrt_two_omega_ = (np.sqrt(2*omega[:,np.newaxis,np.newaxis,np.newaxis,:]))**(-1)
    # f_nophase = f_nophase * inv_sqrt_two_omega_
    # f_phase[:, 0] = f_phase[:,0] * inv_sqrt_two_omega_
    # f_phase[:, 1] = f_phase[:,1] * inv_sqrt_two_omega_

    # TODO_done: omega_qv is not the same order as gkk, it seems I need to modify the code and let epw output right omega too. build a map between omega q and gkk q
    omega = read_omega()  # omega_original.dat is generated here
    q_frac_elph, q_frac_omega, _ = read_qkpoint_And_ab() # read q.dat, k.dat, a.dat, a0.dat and b.dat
    q_elph_map_based_on_q_elph, q_omega_map_based_on_q_elph = q_elph_map_q_omega(q_frac_elph=q_frac_elph, q_frac_omega=q_frac_omega)

    q_omega_map_based_on_q_elph = np.array(q_omega_map_based_on_q_elph, dtype=int)
    omega_new_based_elph_q_order = omega[q_omega_map_based_on_q_elph,:]
    np.savetxt("omega.dat",omega_new_based_elph_q_order.reshape(nq*nmu))

    # final_ep_real, final_ep_imag, final_g2 = (nq,nk,ni,nj,nmu)
    # omega_new_based_elph_q_order = (nq,nmu)

    # for iq in range(nq):
    #     for imu in range(nmu):
    #         if omega_new_based_elph_q_order[iq,imu] > 0:
    #             final_ep_real[iq,:,:,:,imu] = final_ep_real[iq,:,:,:,imu] / np.sqrt(2 * (omega_new_based_elph_q_order[iq,imu]/ry2meV))
    #             final_ep_imag[iq,:,:,:,imu] = final_ep_imag[iq,:,:,:,imu] / np.sqrt(2 * (omega_new_based_elph_q_order[iq,imu]/ry2meV))
    #             final_g2[iq,:,:,:,imu] = final_g2[iq,:,:,:,imu] / (2 * omega_new_based_elph_q_order[iq,imu]/ry2meV)
    #         else:
    #             print('skip: iq: %s, imode: %s, omega(iq,imode):%s'%(iq,imu,omega_new_based_elph_q_order[iq,imu]), ' [meV]')
    #             final_ep_real[iq,:,:,:,imu] = np.zeros_like(final_ep_real[iq,:,:,:,imu])
    #             final_ep_imag[iq,:,:,:,imu] = np.zeros_like(final_ep_imag[iq,:,:,:,imu])
    #             final_g2[iq,:,:,:,imu] = np.zeros_like(final_g2[iq,:,:,:,imu])

    print("gkk/omega is done!")

    n_total = final_g2.shape[0] *\
              final_g2.shape[1] *\
              final_g2.shape[2] *\
              final_g2.shape[3] *\
              final_g2.shape[4]\

    # TODO: memory issue
    # f_phase = np.zeros((n_total, 2))
    # f_phase[:,0] = final_ep_real.reshape(n_total)
    # f_phase[:,1] = final_ep_imag.reshape(n_total)
    # f_nophase = np.sqrt(final_g2.reshape(n_total))

    # todo_done: use h5 or binary format instead of txt
    # np.savetxt('elphmat_phase.dat', f_phase)
    # np.savetxt('elphmat.dat', f_nophase)

    h5_elphmat_phase = h5.File('elphmat_phase.h5','w')
    h5_elphmat_phase.create_dataset('data', (n_total, 2))
    h5_elphmat_phase['data'][:,0] = final_ep_real.reshape(n_total) * ry2meV
    h5_elphmat_phase['data'][:,1] = final_ep_imag.reshape(n_total) * ry2meV
    h5_elphmat_phase.close()

    h5_elphmat = h5.File('elphmat.h5','w')
    h5_elphmat.create_dataset('data',data=np.sqrt(final_g2.reshape(n_total))*ry2meV)
    h5_elphmat.close()



def q_elph_map_q_omega(q_frac_elph, q_frac_omega):
    baseKgrid = q_frac_elph
    f = open('qelph_qomega_qmap.dat', 'w')
    f.write('#This is a map from kgrid to index of gkk and Acv\n')
    f.write('# grid_1 grid_2 grid_3 q_elph q_omega \n')
    for j in range(baseKgrid.shape[0]):
        base_kpoint = baseKgrid[j]
        res = '  %.7f    %.7f    %.7f' % (base_kpoint[0], base_kpoint[1], base_kpoint[2])
        for i in range(2):
            todoKgrid = [q_frac_elph,q_frac_omega][i]
            match = 0
            for k in range(todoKgrid.shape[0]):
                if equivalence_order(move_k_back_to_BZ_1(baseKgrid[j]), move_k_back_to_BZ_1(todoKgrid[k])):
                    match = 1
                    res = res + "    %s" % k
                    continue
            if match == 0:
                res = res + "    -1"
        f.write(res + '\n')
    f.close()

    data = np.loadtxt("qelph_qomega_qmap.dat")
    q_elph_map_based_on_q_elph = data[:,3]
    q_omega_map_based_on_q_elph = data[:,4]

    return q_elph_map_based_on_q_elph,q_omega_map_based_on_q_elph



def read_parameter():
    if not os.path.isfile('ph.out'): raise Exception('Rename your phonon output into "ph.out"')
    if not os.path.isfile('epw.out'): raise Exception('Rename your epw output into "epw.out"')
    if not os.path.isfile('epw.in'): raise Exception('Rename your epw input into "epw.in"')
    if not os.path.isfile('nscf.in'): raise Exception('Rename your epw input into "nscf.in"')
    os.system("grep 'number of atoms' ph.out|head -n 1|awk '{print $5}' > temp_parameter") # number of atom
    os.system("grep 'Number of bands' epw.out |awk '{print $7}'|tr -d ')' >> temp_parameter") # number of bands
    os.system("grep 'Number of total' epw.out |awk '{print $8}'|tr -d ')' >> temp_parameter") # number of total bands
    os.system("grep 'Number of excluded' epw.out |awk '{print $8}'|tr -d ')' >> temp_parameter") # number of excluded bands
    print('WARNING: SEE epw.in for skipped bands and window energy')
    os.system("grep 'number of k points=' epw.out |awk '{print $5}' >> temp_parameter") # number of k points
    os.system("grep 'q(' epw.out | tail -n 1|awk '{print $2}' >> temp_parameter") # number of q points
    os.system("grep 'pools' epw.out |head -n 1|awk '{print $6}' >> temp_parameter") # number of pools
    temp_para = np.loadtxt('temp_parameter')
    [natom,nbnd_calculated, nbnd_total, nbnd_exclude, nk, nq, number_of_epb] = temp_para
    nmu = natom * 3
    print('numu: %s\nnbnd_calculated:%s\nnbnd_total:%s\nnbnd_exclude:%s\nnk:%s\nnq:%s\nnumber_of_epb:%s\n'%(int(nmu),int(nbnd_calculated), int(nbnd_total), int(nbnd_exclude), int(nk), int(nq), int(number_of_epb)))
    return int(nmu),int(nbnd_calculated), int(nbnd_total), int(nbnd_exclude), int(nk), int(nq), int(number_of_epb)

def read_omega():
    """
    :return: omega, this is ordered based on kmesh.pl, which is uniform
    """
    nmode, _, _, _, _, nq, _ = read_parameter()
    os.system("grep 'ik =       1' epw.out -A %s| grep -E '^ +[0-9]+ +[0-9]+ +[0-9]+'|awk '{print $6}' > omega_uniform_q.dat" % (
            nmode + 2))
    omega = np.loadtxt("omega_uniform_q.dat")
    omega = omega.reshape(nq,nmode)
    return omega

def car2frac(A,q_list_frac):
    """
    :param A: [[b1],[b2],[b3]], is reciprocal Matrix
    y_car = A.T @ q_frac ==> q_frac = (A.T)^(-1) @ y_car
    :param q_list_frac: q_list.shape = (N, 3), N is number of q
    :return: q list in fractional system (3, N)
    """
    A = A.T
    A_1 = np.linalg.inv(A)
    b_frac = A_1 @ q_list_frac.T
    q = b_frac.T
    return q


def read_qkpoint_And_ab():
    os.system("grep 'q(' epw.out |awk '{print $6,$7,$8}' > q_car_elph.dat")
    os.system("grep '     iq =' epw.out |awk '{print $5,$6,$7}' > q_frac_omega.dat")
    os.system("grep 'k(' epw.out |awk '{print $5,$6,$7}'|tr -d ')'|tr -d ',' > k_car_elph.dat")
    os.system("grep 'lattice parameter ' epw.out |awk '{print $5}' >a0.dat")
    os.system("grep '            a' epw.out|awk '{print $4,$5,$6}' > a.dat")
    os.system("grep '            b' epw.out|awk '{print $4,$5,$6}' > b.dat")

    b_matrix = np.loadtxt('b.dat')
    q_car_elph = np.loadtxt("q_car_elph.dat")
    q_frac_omega = np.loadtxt("q_frac_omega.dat")
    k_car_elph = np.loadtxt("k_car_elph.dat")

    q_frac_elph = car2frac(b_matrix,q_list_frac=q_car_elph)
    k_frac_elph = car2frac(b_matrix,q_list_frac=k_car_elph)

    np.savetxt("q.dat",q_frac_elph)
    np.savetxt("k.dat",k_frac_elph)
    # We will use q_frac_elph as q order in kkq_map, because it is easier to reorder omega than reordering elph

    return q_frac_elph, q_frac_omega, k_frac_elph



def read_single_epb(prefix,epb_index):
    """
    :param prefix: name of system, e.g.: bn.epbxx
    :param epb_index: index of epb, which is based on pool number
    :param ni: number of final states
    :param nj: number of initial states
    :param nk: number of k points
    :param nq: number of q points
    :param nmu: number of phonon
    :return: it returns ep_real, ep_imag, g2
    # g2 = ep_real**2 + ep_imag**2
    """
    f = h5.File(prefix+'_elph_%s.h5'%epb_index,'r')

    # These three variables el-ph mat in EPW order
    ep_real_epworder = f['elph_nu_real'][()] # (q, nu, k, j, i)
    ep_imag_epworder = f['elph_nu_imag'][()]
    g2_epworder = ep_imag_epworder**2 + ep_real_epworder**2

    # Reshape this to fit order in my EXPH order elph_mat(nq,nk,ni,nj,nmode), this order is actually the same as el-ph matrix on interpolated fine grid
    # Well, anyway, here is how I organize the el-ph matrix in a very simple way (this might have some memory issue if el-ph matrix is very large)

    # (1) (q, nu, k, j ,i) --> (q, k ,nu, j ,i)
    ep_real = np.swapaxes(ep_real_epworder,1,2)
    ep_imag = np.swapaxes(ep_imag_epworder,1,2)
    g2 = np.swapaxes(g2_epworder,1,2)
    # (2) (q,k,nu,j,i) -->  (q,k,i,j,nu)
    ep_real = np.swapaxes(ep_real,2,3)
    ep_imag = np.swapaxes(ep_imag,2,3)
    g2 = np.swapaxes(g2,2,3) # (q,k,j,nu,i)

    ep_real = np.swapaxes(ep_real,3,4)
    ep_imag = np.swapaxes(ep_imag,3,4)
    g2 = np.swapaxes(g2,3,4) # (q,k,j,i,nu)

    ep_real = np.swapaxes(ep_real,2,3)
    ep_imag = np.swapaxes(ep_imag,2,3)
    g2 = np.swapaxes(g2,2,3) # (q,k,i,j,nu)


    return ep_real, ep_imag, g2


def read_single_epb_binary(prefix,epb_index,ni,nj,nk,nq,nmu):
    """
    :param prefix: name of system, e.g.: bn.epbxx
    :param epb_index: index of epb, which is based on pool number
    :param ni: number of final states
    :param nj: number of initial states
    :param nk: number of k points
    :param nq: number of q points
    :param nmu: number of phonon
    :return: it returns ep_real, ep_imag, g2
    # g2 = ep_real**2 + ep_imag**2
    """

    f = FortranFile(prefix+'.epb%s'%epb_index,'r')
    ep = f.read_reals(dtype='float')

    print("epb%s:"%epb_index, ni,nj,nk,nq,nmu)
    print("epb%s:"%epb_index, ep.shape)

    ep_reshap4complex = ep.reshape((nj*ni*nmu*nk*nq,2))


    # These three variables el-ph mat in EPW order
    ep_real_epworder = ep_reshap4complex[:,0].reshape((nj,ni,nk,nmu,nq),order='F')
    ep_imag_epworder = ep_reshap4complex[:,1].reshape((nj,ni,nk,nmu,nq),order='F')
    g2_epworder = ep_imag_epworder**2 + ep_real_epworder**2

    # Reshape this to fit order in my EXPH order elph_mat(nq,nk,ni,nj,nmode), this order is actually the same as el-ph matrix on interpolated fine grid
    # Well, anyway, here is how I organize the el-ph matrix in a very simple way (this might have some memory issue if el-ph matrix is very large)
    # (1) (nj,ni,nk,nmu,nq) --> (ni,nj,nk,nmu,nq)
    ep_real = np.swapaxes(ep_real_epworder,0,1)
    ep_imag = np.swapaxes(ep_imag_epworder,0,1)
    g2 = np.swapaxes(g2_epworder,0,1)
    # (2) (ni,nj,nk,nmu,nq) --> (nk,ni,nj,nmu,nq)
    ep_real = np.swapaxes(ep_real,1,2)
    ep_imag = np.swapaxes(ep_imag,1,2)
    g2 = np.swapaxes(g2,1,2)
    ep_real = np.swapaxes(ep_real,0,1)
    ep_imag = np.swapaxes(ep_imag,0,1)
    g2 = np.swapaxes(g2,0,1)
    # (2) (nk,ni,nj,nmu,nq) --> (nq,nk,ni,nj,nmu)
    ep_real = np.swapaxes(ep_real,3,4)
    ep_imag = np.swapaxes(ep_imag,3,4)
    g2 = np.swapaxes(g2,3,4)
    ep_real = np.swapaxes(ep_real,2,3)
    ep_imag = np.swapaxes(ep_imag,2,3)
    g2 = np.swapaxes(g2,2,3)
    ep_real = np.swapaxes(ep_real,1,2)
    ep_imag = np.swapaxes(ep_imag,1,2)
    g2 = np.swapaxes(g2,1,2)
    ep_real = np.swapaxes(ep_real,0,1)
    ep_imag = np.swapaxes(ep_imag,0,1)
    g2 = np.swapaxes(g2,0,1)

    return ep_real, ep_imag, g2

    # save data
    # f_phase.shape = (n_total, 2)
    # f_phase = np.zeros((n_total,2))
    #
    #
    # f_phase[:,0] = ep_real.reshape(n_total)
    # f_phase[:,1] = ep_imag.reshape(n_total)
    # f_nophase = np.sqrt(g2.reshape(n_total))
    #
    # np.savetxt('elphmat_phase.dat',f_phase)
    # np.savetxt('elphmat.dat',f_nophase)

# def read_epb(prefix,number_of_epb,ni,nj,nk,nq,nmu):

if __name__ == "__main__":
    prefix = 'bn'

    # ni = 10  # initial
    # nj = 10  # final
    # nk = 9
    # nq = 9
    # nmu = 6
    # exclude = 0

    # ep_real, ep_imag, g2 = read_single_epb(prefix=prefix,epb_index=1,ni=ni,nj=nj,nk=nk,nq=nq,nmu=nmu)

    read_epb(prefix=prefix)
