import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

def plot_frame_occupation(i,a,size=1): # TODO: use path instead of a
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
    plt.savefig('occupation.pdf')

def plot_frame_diffusion(i,S, path,Q1=0,Q2=12,Q3=200,Q4=400):
    f = h5.File(path + 'EX_diffusion_evolution.h5', 'r')
    F_nQxy_res = f['data']

    X = np.arange(F_nQxy_res.shape[2])
    Y = np.arange(F_nQxy_res.shape[3])
    XX, YY = np.meshgrid(X, Y)
    fig = plt.figure(figsize=(15, 12.5))
    plt.clf()
    plt.subplot(2, 2, 1)

    plt.contourf(XX, YY, F_nQxy_res[S, Q1, :, :, i],
                 levels=np.linspace(F_nQxy_res[S, Q1, :, :, :].min(), F_nQxy_res[S, Q1, :, :, :].max(), 80))
    plt.title('(n=%s,Q=%s)' % (S, Q1) + 'frame: %s ' % int(i) + 'exciton number: %.2f' % F_nQxy_res[S, Q1, :, :,
                                                                                               i].sum())

    plt.colorbar()

    ######################3
    plt.subplot(2, 2, 2)

    plt.contourf(XX, YY, F_nQxy_res[S, Q2, :, :, i],
                 levels=np.linspace(F_nQxy_res[S, Q2, :, :, :].min(), F_nQxy_res[S, Q2, :, :, :].max(), 80))
    # plt.title(label='t=%s fs'%Time_series[i])
    plt.title('(n=%s,Q=%s)' % (S,Q2) + 'frame: %s ' % int(i) + 'exciton number: %.2f' % F_nQxy_res[S, Q2, :, :,
                                                                                               i].sum())

    plt.colorbar()

    ######################3
    plt.subplot(2, 2, 3)

    plt.contourf(XX, YY, F_nQxy_res[S, Q3, :, :, i],
                 levels=np.linspace(F_nQxy_res[S,Q3, :, :, :].min(), F_nQxy_res[S, Q3, :, :, :].max(), 80))
    # plt.title(label='t=%s fs'%Time_series[i])
    plt.title('(n=%s,Q=%s)' % (S, Q3) + 'frame: %s ' % int(i) + 'exciton number: %.2f' % F_nQxy_res[S, Q3, :, :,
                                                                                               i].sum())

    plt.colorbar()

    ######################3
    plt.subplot(2, 2, 4)

    plt.contourf(XX, YY, F_nQxy_res[S, Q4, :, :, i],
                 levels=np.linspace(F_nQxy_res[S, Q4, :, :, :].min(), F_nQxy_res[S, Q4, :, :, :].max(), 80))
    # plt.title(label='t=%s fs'%Time_series[i])
    plt.title('(n=%s,Q=%s)' % (S, Q4) + 'frame: %s ' % int(i ) + 'exciton number: %.2f' % F_nQxy_res[S, Q4, :, :,
                                                                                               i].sum())
    f.close()
    plt.colorbar()
    plt.savefig('diffusion.png' )

    print('fig saved!')
