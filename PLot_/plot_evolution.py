import numpy as np
from Common.progress import ProgressBar
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time
import h5py as h5
import sys

def plot_diff_evolution(F_nQxy_res,nX=80,nY=80,n=2,T_total=40,delta_T=0.02,play_interval=2, path='../',saveformat='htm',Q1=0,Q2=12,Q3=200,Q4=400):
    """
    :param nX: number of discrete X
    :param nY: number of discrete Y
    :param n: number of EXCITON state
    :param T_total: total time in [fs]
    :param delta_T: delta_T in [fs]
    :param play_interval: plot evolution in every "plat_interval" [fs]
    :param path:
    :return:
    """
    nT = int(T_total / delta_T)
    X = np.arange(nX)
    Y = np.arange(nY)
    XX, YY = np.meshgrid(X, Y)
    fig = plt.figure(figsize=(15, 12.5))

    # f = h5.File(path+'EX_diffusion_evolution.h5','r')
    # F_nQxy_res = f['data'][()]
    # f.close()
    print('F_nQxy_res has been read and size of it is: %.2f MB'%(sys.getsizeof(F_nQxy_res)/1024/1024))

    def animate(i):
        plt.clf()
        plt.subplot(2, 2, 1)

        plt.contourf(XX, YY, F_nQxy_res[n, Q1, :, :, i],
                     levels=np.linspace(F_nQxy_res[n, Q1, :, :, :].min(), F_nQxy_res[n, Q1, :, :, :].max(), 80))
        plt.title('(n=%s,Q=%s)' % (n, Q1) + 't=%s fs' % int(i * delta_T) + 'exciton number: %.2f' % F_nQxy_res[n, Q1, :, :,
                                                                                                   i].sum())

        plt.colorbar()

        ######################3
        plt.subplot(2, 2, 2)

        plt.contourf(XX, YY, F_nQxy_res[n, Q2, :, :, i],
                     levels=np.linspace(F_nQxy_res[n,Q2, :, :, :].min(), F_nQxy_res[n, Q2, :, :, :].max(), 80))
        # plt.title(label='t=%s fs'%Time_series[i])
        plt.title('(n=%s,Q=%s)' % (n, Q2) + 't=%s fs' % int(i * delta_T) + 'exciton number: %.2f' % F_nQxy_res[n, Q2, :, :,
                                                                                                   i].sum())

        plt.colorbar()

        ######################3
        plt.subplot(2, 2, 3)

        plt.contourf(XX, YY, F_nQxy_res[n, Q3, :, :, i],
                     levels=np.linspace(F_nQxy_res[n,Q3, :, :, :].min(), F_nQxy_res[n, Q3, :, :, :].max(), 80))
        # plt.title(label='t=%s fs'%Time_series[i])
        plt.title('(n=%s,Q=%s)' % (n, Q3) + 't=%s fs' % int(i * delta_T) + 'exciton number: %.2f' % F_nQxy_res[n, Q3, :, :,
                                                                                                   i].sum())

        plt.colorbar()

        ######################3
        plt.subplot(2, 2, 4)

        plt.contourf(XX, YY, F_nQxy_res[n, Q4, :, :, i],
                     levels=np.linspace(F_nQxy_res[n, Q4, :, :, :].min(), F_nQxy_res[n, Q4, :, :, :].max(), 80))
        # plt.title(label='t=%s fs'%Time_series[i])
        plt.title('(n=%s,Q=%s)' % (n, Q4) + 't=%s fs' % int(i * delta_T) + 'exciton number: %.2f' % F_nQxy_res[n, Q4, :, :,
                                                                                                   i].sum())

        plt.colorbar()


    ani = animation.FuncAnimation(fig, animate, np.arange(0, nT, int(play_interval / delta_T)), interval=7)

    if saveformat:
        print("saving diffusion")
        ani.save(path+'diffusion.'+saveformat)
        print("saving diffusion done!")

    return ani # see why I return ani here: https://stackoverflow.com/questions/21099121/python-matplotlib-unable-to-call-funcanimation-from-inside-a-function