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


class Solver_of_only_Q_space(InitialInformation):
    def __init__(self,path='../',degaussian=0.03,T=500,initial_occupation=0.5, delta_T =1, T_total=3000, play_interval = 50):
        super(Solver_of_only_Q_space,self).__init__(path,degaussian,T)
        # Initialize F_nQ and N_vq
        # E_nQ.shape = (n,Q)
        self.F_nQ = BE(omega=self.get_E_nQ(), T=self.T)
        self.F_nQ[2,0] = initial_occupation
        self.N_vq = BE(omega=self.get_E_vq(),T=T)
        self.N_vq[0:3,0] = np.array([0,0,0])
        self.Delta_positive, self.Delta_negative = self.Construct_Delta()

        # Plot Setting:
        self.play_interval = play_interval
        self.T_total = T_total
        self.delta_T = delta_T
        self.nT = int(T_total/delta_T)

        # Res Container:
        self.F_nQ_res = np.zeros((self.n, self.Q, self.nT))
        self.exciton_number = np.zeros(self.nT)
        self.dfdt_sum_res = np.zeros(self.nT)
        self.damping_term = np.zeros((self.n, self.Q))



        self.Q_exciton = np.arange(self.Q) * np.ones((self.n, self.Q))
        self.energy = self.get_E_nQ()
        self.Time_series = np.arange(0,self.T_total,self.play_interval)
        print("Initialized information has been loaded")

    def __update_F_nQq(self,F_nQ_last):
        """
        :param F_nQ:
        :return: F_nQq.shape = (n,Q,q)
        """
        Q_acv_back2_kmap = list(map(int, list(self.kmap[:, 3])))
        F_nQq = np.zeros((F_nQ_last.shape[0], F_nQ_last.shape[1], F_nQ_last.shape[1]))
        for i_Q_kmap in range(self.Q):
            for i_q_kmap in range(self.q):
                F_nQq[:, i_Q_kmap, i_q_kmap] = F_nQ_last[:,
                                               Q_acv_back2_kmap.index(self.Qplusq_2_QPrimeIndexAcv(i_Q_kmap, i_q_kmap))]
        return F_nQq

    def __rhs_Fermi_Goldenrule(self,F_nQ_last):
        F_nQq = self.__update_F_nQq(F_nQ_last)
        F_abs = np.einsum('np,vq,npq->npqv',F_nQ_last, self.N_vq, 1 + F_nQq) \
                - np.einsum('np,vq,npq->npqv', 1 + F_nQ_last, 1 + self.N_vq, F_nQq)
        F_em = np.einsum('np,vq,npq->npqv', F_nQ_last, 1 + self.N_vq, 1 + F_nQq) \
               - np.einsum('np,vq,npq->npqv', 1 + F_nQ_last, self.N_vq, F_nQq)
        dFdt = (np.einsum('pqnmv,nmvpq,npqv->np', self.gqQ_mat, self.Delta_positive, F_abs) + np.einsum(
            'pqnmv,nmvpq,npqv->np', self.gqQ_mat, self.Delta_negative, F_em))
        return -1*(np.pi * 2)/(self.h_bar * self.Q) * dFdt

    def solve_it(self):
        progress = ProgressBar(self.nT, fmt=ProgressBar.FULL)
        for it in range(self.nT):
            self.damping_term[:, 0] = self.F_nQ[:, 0]
            progress.current += 1
            progress()
            self.F_nQ_res[:, :, it] = self.F_nQ
            dfdt = self.__rhs_Fermi_Goldenrule(self.F_nQ)
            error_from_nosymm = dfdt.sum() / (dfdt.shape[0] * dfdt.shape[1])

            self.F_nQ = self.F_nQ\
                        + (dfdt - error_from_nosymm) * self.delta_T\
                        - self.damping_term * self.delta_T * 0.

            # some debugging
            self.exciton_number[it] = self.F_nQ.sum()
            self.dfdt_sum_res[it] = dfdt.sum()
            # print(dfdt[2,0])

    def write_occupation_evolution(self):
        f = h5.File(self.path+'EX_band_evolution.h5','w')
        f.create_dataset('data',data=self.F_nQ_res)
        f.close()
        print('EX_band_evolution.h5 has been written')

    def plot(self,saveformat=None):
        def animate(i):
            plt.clf()
            plt.scatter(self.Q_exciton, self.energy, s=self.F_nQ_res[:, :, i] * 10000)
            plt.title(label='t=%s fs' % self.Time_series[i] + ' total_exciton: %.1f'%self.F_nQ_res[:,:,i].sum())
            plt.xlabel("Q")
            plt.ylabel("Energy")
            plt.xlim([self.Q_exciton.min()-1 , self.Q_exciton.max()+1])
            plt.ylim([self.energy.min()-1, self.energy.max()+1])
            plt.show()

        fig = plt.figure()
        # animate(3)
        ani = animation.FuncAnimation(fig, animate, np.arange(self.T_total // self.play_interval), interval=10)
        # print('plot done')
        if saveformat:
            ani.save(self.path+'occupation.'+saveformat)
        return ani

if __name__ == "__main__":
    a = Solver_of_only_Q_space(delta_T=1,T_total=1000,play_interval=10,path='../')
    a.solve_it()
    a.write_occupation_evolution()
    ani = a.plot(saveformat=None)
