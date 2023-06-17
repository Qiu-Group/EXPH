import sys
import numpy as np
import h5py as h5
from IO.IO_gkk import read_omega, read_gkk
from IO.IO_acv import read_Acv, read_Acv_exciton_energy
from Common.common import move_k_back_to_BZ_1
from Common.distribution import Dirac_1, BE
from Common.progress import ProgressBar
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IO.IO_common import read_kmap, read_lattice
from IO.IO_common import construct_kmap, read_kmap, read_bandmap, readkkqQ
from ELPH.EX_PH_Boltzman_Class import InitialInformation
import cupy as cp
import time
from PLot_.plot_frame_evolution import plot_frame_occupation

class Solver_of_only_Q_space(InitialInformation):
    def __init__(self,path='../',degaussian=0.03,T=500,initial_occupation=0.5, delta_T =1, T_total=3000, play_interval = 50,initial_S=2,initial_Q=0,initial_Gaussian_Braod=1,
                 high_symm="0.0 0.0 0.0, 0.75 0.75 0, 0.75 0.0 0.0, 0.0 0.0 0.0",onGPU=True):
        super(Solver_of_only_Q_space,self).__init__(path=path,deguassian=degaussian,T=T,initial_S=initial_S,initial_Q=initial_Q,initial_Gaussian_Braod=initial_Gaussian_Braod,high_symm=high_symm,onGPU=onGPU)
        # Initialize F_nQ and N_vq
        # E_nQ.shape = (n,Q)
        self.F_nQ = BE(omega=self.get_E_nQ(), T=self.T)
        self.F_nQ[self.initial_S,self.initial_Q] = initial_occupation
        self.N_vq = BE(omega=self.get_E_vq(),T=T)
        self.N_vq[0:3,0] = np.array([0,0,0])
        self.Delta_positive, self.Delta_negative = self.Construct_Delta()
        # self.Delta_positive, self.Delta_negative = np.ones_like(self.Delta_positive), np.ones_like(self.Delta_negative) # TODO: debug!!!!!
        self.onGPU =onGPU
        # Plot Setting:
        self.play_interval = play_interval
        self.T_total = T_total
        self.delta_T = delta_T
        self.nT = int(T_total/delta_T)
        self.high_symm = high_symm
        self.high_symms_path_index_list_in_kmap = self.get_highsymmetry_index_kmap()

        # Res Container:
        self.dfdt_res = np.zeros((self.n, self.Q, self.nT))
        self.F_nQ_res = np.zeros((self.n, self.Q, self.nT))
        self.exciton_number = np.zeros(self.nT)
        self.dfdt_sum_res = np.zeros(self.nT)
        self.damping_term = np.zeros((self.n, self.Q))

        self.Qq_2_Qpr_res = self.Qq_2_Qpr_kmap()

        # self.Q_exciton = np.arange(self.Q) * np.ones((self.n, self.Q))
        self.Q_exciton = np.arange(len(self.high_symms_path_index_list_in_kmap)) * np.ones((self.n, len(self.high_symms_path_index_list_in_kmap)))
        self.energy = self.get_E_nQ()
        self.Time_series = np.arange(0,self.T_total,self.play_interval)
        print("Initialized information has been loaded")

    def Qq_2_Qpr_kmap(self):
        """
        start from 0
        """
        Q_acv_back2_kmap = list(map(int, list(self.kmap[:, 3])))
        Qq_2_Qpr_res = np.zeros((self.Q,self.q),dtype=int)
        for i_Q_kmap in range(self.Q):
            for i_q_kmap in range(self.q):
                Qq_2_Qpr_res[i_Q_kmap, i_q_kmap] = Q_acv_back2_kmap.index(self.Qplusq_2_QPrimeIndexAcv(i_Q_kmap, i_q_kmap))
        return Qq_2_Qpr_res


    def update_F_nQq(self,F_nQ_last):
        """
        :param F_nQ:
        :return: F_nQq.shape = (n,Q,q)
        """

        # Q_acv_back2_kmap = list(map(int, list(self.kmap[:, 3])))
        # F_mQq = np.zeros((F_nQ_last.shape[0], F_nQ_last.shape[1], F_nQ_last.shape[1]))
        # for i_Q_kmap in range(self.Q):
        #     for i_q_kmap in range(self.q):
        #         F_mQq[:, i_Q_kmap, i_q_kmap] = F_nQ_last[:,
        #                                        Q_acv_back2_kmap.index(self.Qplusq_2_QPrimeIndexAcv(i_Q_kmap, i_q_kmap))] # TODO: You can actually doing this for one time!

        # F_mQq[:,:,:] = F_nQ_last[:,Qq_2_Qpr_res]

        # return F_mQq
        return F_nQ_last[:,self.Qq_2_Qpr_res]

    def __rhs_Fermi_Goldenrule_GPU(self,F_nQ_last):
        # t0 = time.time()
        F_mQq = self.update_F_nQq(F_nQ_last)
        # print('  (1) time for updating F_mQq',time.time() - t0,'s')

        # load data from CPU to GPU memory: >>>>>>>>>>
        # t0 = time.time()
        F_mQq = cp.array(F_mQq)
        self.N_vq = cp.array(self.N_vq)
        self.gqQ_mat = cp.array(self.gqQ_mat)
        self.Delta_positive = cp.array(self.Delta_positive)
        self.Delta_negative = cp.array(self.Delta_negative)
        F_nQ_last = cp.array(F_nQ_last)
        # print('  (2) time for loading data from cpu to gpu', time.time() - t0, 's') # TODO: change index order below to test if it will become faster, like: 'np,vq,mpq->nmpqv' ==> 'np,vq,mpq->nmvpq
        #-----------------------------------<<<<<<<<<<
        # t0 = time.time()
        F_abs = cp.einsum('np,vq,mpq->nmpqv',F_nQ_last, self.N_vq, 1 + F_mQq,optimize='greedy') \
                - cp.einsum('np,vq,mpq->nmpqv', 1 + F_nQ_last, 1 + self.N_vq, F_mQq,optimize='greedy')
        # F_em = np.einsum('np,vq,npq->npqv', F_nQ_last, 1 + self.N_vq, 1 + F_nQq) \
        F_em =  - cp.einsum('np,vq,mpq->nmpqv', 1 + F_nQ_last, self.N_vq, F_mQq,optimize='greedy')\
                + cp.einsum('np,vq,mpq->nmpqv', F_nQ_last, 1 + self.N_vq, 1 + F_mQq,optimize='greedy')
        dFdt = cp.einsum('pqnmv,nmvpq,nmpqv->np', self.gqQ_mat, self.Delta_positive, F_abs,optimize='greedy')  \
                + cp.einsum('pqnmv,nmvpq,nmpqv->np', self.gqQ_mat, self.Delta_negative, F_em,optimize='greedy')
        # <<<--------for debug
        # print('  (3) time for GPU calculation', time.time() - t0, 's')

        # t0 = time.time()
        dFdt = cp.asnumpy(dFdt)
        # print('  (4) time for transfer data to CPU', time.time() - t0, 's')
        return -1*(np.pi * 2)/(self.h_bar * self.Q) * dFdt

    def __rhs_Fermi_Goldenrule_CPU(self,F_nQ_last):
        # t0 = time.time()
        F_mQq = self.update_F_nQq(F_nQ_last)
        F_abs = np.einsum('np,vq,mpq->nmpqv',F_nQ_last, self.N_vq, 1 + F_mQq,optimize='greedy') \
                - np.einsum('np,vq,mpq->nmpqv', 1 + F_nQ_last, 1 + self.N_vq, F_mQq,optimize='greedy')
        F_em =  - np.einsum('np,vq,mpq->nmpqv', 1 + F_nQ_last, self.N_vq, F_mQq,optimize='greedy')\
                + np.einsum('np,vq,mpq->nmpqv', F_nQ_last, 1 + self.N_vq, 1 + F_mQq,optimize='greedy')
        dFdt = np.einsum('pqnmv,nmvpq,nmpqv->np', self.gqQ_mat, self.Delta_positive, F_abs,optimize='greedy')  \
                + np.einsum('pqnmv,nmvpq,nmpqv->np', self.gqQ_mat, self.Delta_negative, F_em,optimize='greedy')
        # print('  (4) time for transfer data to CPU', time.time() - t0, 's')
        return -1*(np.pi * 2)/(self.h_bar * self.Q) * dFdt

    def solve_it(self):
        t0 = time.time()
        progress = ProgressBar(self.nT, fmt=ProgressBar.FULL)

        if self.onGPU:
            print('[GPU acceleration] ON')
        else:
            print('[GPU acceleration] OFF')

        for it in range(self.nT):
            progress.current += 1
            progress()

            # --->>>>> solving <<<<-----
            self.damping_term[:, 0] = self.F_nQ[:, 0]
            self.F_nQ_res[:, :, it] = self.F_nQ
            if self.onGPU:
                dfdt = self.__rhs_Fermi_Goldenrule_GPU(self.F_nQ)
            else:
                dfdt = self.__rhs_Fermi_Goldenrule_CPU(self.F_nQ)
            self.dfdt_res[:,:,it] = dfdt # TODO: debugging

            self.F_nQ = self.F_nQ + dfdt * self.delta_T \
                        - self.damping_term * self.delta_T * 0.0

            self.exciton_number[it] = self.F_nQ.sum()
            self.dfdt_sum_res[it] = dfdt.sum()
            # --->>>>> solving done<<<<-----

        print('\n\ntime for solving occupation ODE:' , time.time() - t0, 's')
    def write_occupation_evolution(self):
        f = h5.File(self.path+'EX_band_evolution_.h5','w')
        f.create_dataset('data',data=self.F_nQ_res)
        f.close()
        print('EX_band_evolution.h5 has been written')

    def plot(self,saveformat=None,readfromh5=False):
        if not readfromh5:
            self.F_nQ_res = self.F_nQ_res
        else:
            print('reding band evolution data')
            f = h5.File(self.path + 'EX_band_evolution_.h5', 'r')
            self.F_nQ_res = f['data'][()]
            print('size of occupation.h5: %.2f MB'%(sys.getsizeof(self.F_nQ_res)/1024/1024))
            f.close()
        def animate(i):

            plt.clf()
            plt.scatter(self.Q_exciton, self.energy[:,self.high_symms_path_index_list_in_kmap], s=np.sqrt(self.F_nQ_res[:, self.high_symms_path_index_list_in_kmap, i])**1.5 * 1000, color='r')
            plt.title(label='t=%s fs' % int(i * self.delta_T) + ' total_exciton: %.1f'%self.F_nQ_res[:,:,i].sum())
            plt.xlabel("Q")
            plt.ylabel("Energy")
            plt.xlim([self.Q_exciton.min()-1 , self.Q_exciton.max()+1])
            plt.ylim([self.energy.min()-0.1, self.energy.max()+0.1])
            # plt.ylim(1.0,1.5)
            plt.show()

        fig = plt.figure()
        # animate(3)
        ani = animation.FuncAnimation(fig, animate,np.arange(0, self.nT, int(self.play_interval / self.delta_T)), interval=10)
        # print('plot done')
        if saveformat:
            ani.save(self.path+'occupation.'+saveformat)
        return ani


if __name__ == "__main__":
    a = Solver_of_only_Q_space(degaussian=0.02,delta_T=1,T_total=1000,play_interval=5,path='../',
                               initial_S=0,initial_Q=0,initial_Gaussian_Braod=1,
                               high_symm="0.0 0.0 0.0 ,0.3333333 0.33333333 0.0, 0.5 0.0 0, 0.0 0.0 0.0",
                               initial_occupation=2,T=100, onGPU=False)
    a.solve_it()
    a.write_occupation_evolution()
    ani = a.plot(saveformat=None,readfromh5=True)
    print('done!')

    #plot_frame_occupation()