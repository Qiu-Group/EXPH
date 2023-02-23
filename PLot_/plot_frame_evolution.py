import numpy as np
import matplotlib.pyplot as plt


def plot_frame_occupation(i,a,size=1):
    plt.figure(figsize=(20,12))
    plt.scatter(a.Q_exciton, a.energy[:, a.high_symms_path_index_list_in_kmap],
                s=np.sqrt(a.F_nQ_res[:, a.high_symms_path_index_list_in_kmap, i]) ** 1.5 * 1000*size, color='r')
    plt.title(label='t=%s fs' % int(i * a.delta_T) + ' total_exciton: %.1f' % a.F_nQ_res[:, :, i].sum())
    plt.xlabel("Q")
    plt.ylabel("Energy")
    plt.xlim([a.Q_exciton.min() - 1, a.Q_exciton.max() + 1])
    # plt.ylim([self.energy.min()-0.1, self.energy.max()+0.1])
    plt.ylim(1.15, 1.5)
    plt.show()

def plot_frame_diffusion(i,a,S):
    nT = int(a.T_total / a.delta_T)
    X = np.arange(a.nX)
    Y = np.arange(a.nY)
    XX, YY = np.meshgrid(X, Y)
    fig = plt.figure(figsize=(15, 12.5))
    plt.clf()
    plt.subplot(2, 2, 1)

    plt.contourf(XX, YY, a.F_nQxy_res[S, 0, :, :, i],
                 levels=np.linspace(a.F_nQxy_res[S, 0, :, :, 0].min(), a.F_nQxy_res[S, 0, :, :, 0].max()+0.00000000001, 80))
    plt.title('(n=%s,Q=%s)' % (S, 0) + 't=%s fs' % int(i * a.delta_T) + 'exciton number: %.2f' % a.F_nQxy_res[S, 0, :, :,
                                                                                               i].sum())

    plt.colorbar()

    ######################3
    plt.subplot(2, 2, 2)

    plt.contourf(XX, YY, a.F_nQxy_res[S, 1, :, :, i],
                 levels=np.linspace(a.F_nQxy_res[S, 1, :, :, :].min(), a.F_nQxy_res[S, 1, :, :, :].max(), 80))
    # plt.title(label='t=%s fs'%Time_series[i])
    plt.title('(n=%s,Q=%s)' % (S, 1) + 't=%s fs' % int(i * a.delta_T) + 'exciton number: %.2f' % a.F_nQxy_res[S, 1, :, :,
                                                                                               i].sum())

    plt.colorbar()

    ######################3
    plt.subplot(2, 2, 3)

    plt.contourf(XX, YY, a.F_nQxy_res[S, 2, :, :, i],
                 levels=np.linspace(a.F_nQxy_res[S, 2, :, :, :].min(), a.F_nQxy_res[S, 2, :, :, :].max(), 80))
    # plt.title(label='t=%s fs'%Time_series[i])
    plt.title('(n=%s,Q=%s)' % (S, 2) + 't=%s fs' % int(i * a.delta_T) + 'exciton number: %.2f' % a.F_nQxy_res[S, 2, :, :,
                                                                                               i].sum())

    plt.colorbar()

    ######################3
    plt.subplot(2, 2, 4)

    plt.contourf(XX, YY, a.F_nQxy_res[S, 3, :, :, i],
                 levels=np.linspace(a.F_nQxy_res[S, 3, :, :, :].min(), a.F_nQxy_res[S, 3, :, :, :].max(), 80))
    # plt.title(label='t=%s fs'%Time_series[i])
    plt.title('(n=%s,Q=%s)' % (S, 3) + 't=%s fs' % int(i * a.delta_T) + 'exciton number: %.2f' % a.F_nQxy_res[S, 3, :, :,
                                                                                               i].sum())

    plt.colorbar()
