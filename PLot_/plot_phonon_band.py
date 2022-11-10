from IO.IO_common import write_loop
from Common.common import frac2carte
from Common.h5_status import check_h5_tree
import numpy as np
import h5py as h5
from IO.IO_common import read_kmap, read_lattice
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from ELPH.EX_PH_inteqp import OMEGA_inteqp_Q, omega_inteqp_q
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def plot_phonon_band_inteqp(V_index=0, interposize=12, path='./', outfilename = 'phonon_band.dat'):
    # V_index = nphonon - 1
    bvec = read_lattice('b',path)
    size = int(interposize ** 2)
    [qxx_new, qyy_new, omega_res] =omega_inteqp_q(interpo_size=interposize,new_q_out=True,path=path)

    res = np.zeros((size, 4))
    res[:,0] = qxx_new.flatten()
    res[:,1] = qyy_new.flatten()
    res[:,3] = omega_res[V_index].flatten()
    res[:,:3] = frac2carte(bvec,res[:,:3])

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(res[:,0].reshape((interposize,interposize)), res[:,1].reshape((interposize,interposize)), res[:,3].reshape((interposize,interposize)), cmap=cm.cool)
    plt.show()
    np.savetxt(outfilename, res)


if __name__ == "__main__":
    plot_phonon_band_inteqp(V_index=2, interposize=120,path='../')