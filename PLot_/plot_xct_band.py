from IO.IO_common import write_loop
from Common.common import frac2carte
from Common.h5_status import check_h5_tree
import numpy as np
import h5py as h5
from IO.IO_common import read_kmap, read_lattice
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from ELPH.EX_PH_inteqp import OMEGA_inteqp_Q
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# S = 0 # index of exciton state
# path = '../Acv.h5'
# outfilename = 'exciton.dat'

def plot_exciton_band_inteqp(nS=1, interposize=12, path='./', outfilename = 'exciton_band.dat'):
    S_index = nS - 1
    bvec = read_lattice('b',path)
    size = int(interposize ** 2)
    [Qxx_new, Qyy_new, OMEGA_res] = OMEGA_inteqp_Q(interpo_size=interposize,new_Q_out=True,path=path)

    res = np.zeros((size, 4))
    res[:,0] = Qxx_new.flatten()
    res[:,1] = Qyy_new.flatten()
    res[:,3] = OMEGA_res[S_index].flatten()
    res[:,:3] = frac2carte(bvec,res[:,:3])

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(res[:,0].reshape((interposize,interposize)), res[:,1].reshape((interposize,interposize)), res[:,3].reshape((interposize,interposize)), cmap=cm.cool)
    plt.show()
    np.savetxt(outfilename, res)


# todo: this is deprecated 10/31/2022:
def plot_exciton_band_nointeqp(nS=2, path= './',outfilename = 'exciton_band.dat'):
    """
    :param nS: quantum number of exciton
    :param path: path of Acv.h5
    :param outfilename: path of output file
    """
    S_index = nS - 1
    f = h5.File(path+'Acv.h5','r')
    Seigenval = f['exciton_data/eigenvalues'][()]
    Qgrid = f['exciton_header/kpoints/Qpt_coor'][()]
    bvec = read_lattice('b',path)

    res = np.zeros((Qgrid.shape[0],4))
    for iQ in range(Qgrid.shape[0]):
        # progress.current += 1
        # progress()

        res[iQ, :3] = frac2carte(bvec,Qgrid[iQ])  # give out bohr lattice in reciprocal space
        # res[iQ,:3] = Qgrid[iQ]
        res[iQ, 3] = Seigenval[iQ,S_index]

        write_loop(loop_index=iQ, filename=outfilename, array=res[iQ])

    np.savetxt(outfilename, res)
    # X = res[:,0].reshape(12,12)
    # Y = res[:,1].reshape(12,12)
    # Z = res[:,-1].reshape(12,12)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    #
    # ax.plot_surface(X, Y, Z, cmap="autumn_r", lw=0.5, rstride=1, cstride=1, alpha=0.5)
    # ax.contour(X, Y, Z, 10, lw=3, cmap="autumn_r", linestyles="solid", offset=-1)
    # ax.contour(X, Y, Z, 10, lw=3, colors="k", linestyles="solid")
    # plt.show()

if __name__ == "__main__":
    # plot_exciton_band_nointeqp(1, path='../')
    plot_exciton_band_inteqp(nS=0,path='../',interposize=120)