from ELPH.EX_PH_scat import Gamma_scat_test_nointeqp, Gamma_scat_low_efficiency_inteqp, interpolation_check_for_Gamma_calculation
import numpy as np
from IO.IO_gkk import read_omega, read_gkk
from IO.IO_acv import read_Acv, read_Acv_exciton_energy
from IO.IO_common import read_bandmap, read_kmap, read_lattice,construct_kmap
from Common.common import frac2carte
from Common.progress import ProgressBar
from IO.IO_common import write_loop
from mpi4py import MPI
import  time

def para_Exciton_Life(n_ext_acv_index=0, T=100, degaussian = 0.001, interposize=4,
                 path='./',mute=False):
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
    start_time = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

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

    # this is good enough for parallel!!!
    # we can directly add res of every procs togather, then we can have final collected res_total
    res = np.zeros((kmap.shape[0],4))

    interpolation_check_res = interpolation_check_for_Gamma_calculation(interpo_size=interposize,
                                                                        path=path,
                                                                        mute=True)
    for Q_kmap in range(kmap.shape[0]):
        # progress.current += 1
        # progress()
        #todo: check why this is so slow!
        res[Q_kmap,:3] = frac2carte(bvec,kmap[Q_kmap, 0:3]) # give out bohr lattice in reciprocal space
        # res[Q_kmap, 3] = 1/Gamma_scat_test_nointeqp(Q_kmap=Q_kmap, n_ext_acv_index=n_ext_acv_index, T=T, degaussian=degaussian,muteProgress=mute,path=path)
        res[Q_kmap, 3] = 1 / Gamma_scat_low_efficiency_inteqp(Q_kmap=Q_kmap,
                                                              n_ext_acv_index=n_ext_acv_index,
                                                              T=T,
                                                              degaussian=degaussian,
                                                              muteProgress=mute,
                                                              path=path,
                                                              interpolation_check_res=interpolation_check_res)



        # if rank == 0:
        #     print("write!")
        #     write_loop(loop_index=Q_kmap,filename=outfilename,array=res[Q_kmap])


    if rank == 0:
        np.savetxt(outfilename, res)
        end_time = time.time()
        print("the running time is: %.3f s" % (end_time - start_time))
        return res
    # np.savetxt(outfilename,res)
    # os.remove('./'+'TEMP-' + outfilename)

if __name__ == "__main__":
    res = para_Exciton_Life(path='../', mute=True)