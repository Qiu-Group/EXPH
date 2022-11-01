from IO.IO_common import write_loop
from Common.common import frac2carte
from Common.h5_status import check_h5_tree
from ELPH.EX_PH_mat import gqQ, gqQ_inteqp_q_nopara
from IO.IO_common import read_kmap, read_lattice
from IO.IO_acv import read_Acv
from IO.IO_gkk import read_gkk
from IO.IO_common import read_bandmap, read_kmap,construct_kmap
from Common.progress import ProgressBar
from IO.IO_common import read_kmap, read_lattice
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import numpy as np
from Parallel.Para_EX_PH_lifetime_all_Q import para_Exciton_Life_standard
from ELPH.EX_PH_scat import interpolation_check_for_Gamma_calculation
from ELPH.EX_PH_inteqp import dispersion_inteqp_complete, kgrid_inteqp_complete
from Common.inteqp import interqp_2D
import os

def plot_ex_lifetime_inteqp(n_ext_acv_index=0, T=100, degaussian = 0.001, path='./',mute=True, interposize_for_LifetimeGamma=12, interposize_for_Lifetime = 12, start_from_zero = True):
    bvec = read_lattice('b', path)
    if start_from_zero:
        print('calculating lifetime!')

        res = para_Exciton_Life_standard(n_ext_acv_index=n_ext_acv_index,T=T,degaussian=degaussian,path=path,mute=mute, interposize=interposize_for_LifetimeGamma,write=False)
        print(res)
        size_co = int(np.sqrt(res.shape[0]))
        print('getting Qxx adn Qyy')
        [grdi_q_gqQ_res,_,_,_] = interpolation_check_for_Gamma_calculation(interpo_size=interposize_for_Lifetime, path=path, mute=False)
        Qxx = grdi_q_gqQ_res[:,0].reshape((interposize_for_Lifetime,interposize_for_Lifetime))
        Qyy = grdi_q_gqQ_res[:,1].reshape((interposize_for_Lifetime,interposize_for_Lifetime))

        print('getting lifetime_coarse')
        [_, _, lifetime_res_temp0] = [res[:, 0].reshape((size_co, size_co)), res[:, 1].reshape((size_co, size_co)),
                                res[:, 3].reshape((size_co, size_co))]

        lifetime_res_temp = dispersion_inteqp_complete(lifetime_res_temp0)
        # print('')
        interposize_for_Lifetime += 1

        # Qxx_temp = kgrid_inteqp_complete(Qxx)
        # Qyy_temp = kgrid_inteqp_complete(Qyy)
        lifetime_res = interqp_2D(lifetime_res_temp, interpo_size=interposize_for_Lifetime)[:interposize_for_Lifetime-1, :interposize_for_Lifetime-1]
        print('Plotting')
        # interposize_for_Lifetime -= 1
        interposize_for_Lifetime -= 1
        res = np.zeros((interposize_for_Lifetime**2, 4))

        res[:,0] = Qxx.flatten()
        res[:,1] = Qyy.flatten()
        res[:,3] = lifetime_res.flatten()
        res[:,:3] = frac2carte(bvec,res[:,:3])

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # surf = ax.plot_surface(Qxx,Qyy,lifetime_res, cmap=cm.cool)
        surf = ax.plot_surface(res[:, 0].reshape((interposize_for_Lifetime,interposize_for_Lifetime)), res[:, 1].reshape((interposize_for_Lifetime, interposize_for_Lifetime)),
                               res[:, 3].reshape((interposize_for_Lifetime, interposize_for_Lifetime)), cmap=cm.cool)

        plt.show()
        # res = np.array([Qxx.flatten(),Qyy.flatten,np.zeros(int(interposize_for_Lifetime-1)**2),lifetime_res.flatten()]).T
        np.savetxt('exciton_lifetime.dat',res)
        return res
    else:
        print("from 2")
        res = np.loadtxt('exciton_lifetime.dat')
        print(res)
        size = int(np.sqrt(res.shape[0]))

        [grdi_q_gqQ_res, _, _, _] = interpolation_check_for_Gamma_calculation(interpo_size=interposize_for_Lifetime,
                                                                              path=path, mute=False)
        Qxx = grdi_q_gqQ_res[:,0].reshape((interposize_for_Lifetime,interposize_for_Lifetime))
        Qyy = grdi_q_gqQ_res[:,1].reshape((interposize_for_Lifetime,interposize_for_Lifetime))

        lifetime_res_temp = dispersion_inteqp_complete(res[:, 3].reshape((size,size)))
        interposize_for_Lifetime += 1

        lifetime_res = interqp_2D(lifetime_res_temp, interpo_size=interposize_for_Lifetime)[:interposize_for_Lifetime-1, :interposize_for_Lifetime-1]
        print('Plotting')
        # interposize_for_Lifetime -= 1
        interposize_for_Lifetime -= 1
        res = np.zeros((interposize_for_Lifetime**2, 4))

        res[:,0] = Qxx.flatten()
        res[:,1] = Qyy.flatten()
        res[:,3] = lifetime_res.flatten()
        res[:,:3] = frac2carte(bvec,res[:,:3])

        # surf = ax.plot_surface(Qxx,Qyy,lifetime_res, cmap=cm.cool)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # surf = ax.plot_surface(Qxx,Qyy,lifetime_res, cmap=cm.cool)
        surf = ax.plot_surface(res[:, 0].reshape((interposize_for_Lifetime,interposize_for_Lifetime)), res[:, 1].reshape((interposize_for_Lifetime, interposize_for_Lifetime)),
                               res[:, 3].reshape((interposize_for_Lifetime, interposize_for_Lifetime)), cmap=cm.cool)

        plt.show()
        np.savetxt('exciton_lifetime_new.dat', res)

        return  res


if __name__ == "__main__":
    res = plot_ex_lifetime_inteqp(path='../',start_from_zero=False, mute=False, interposize_for_Lifetime=132,interposize_for_LifetimeGamma=48)
