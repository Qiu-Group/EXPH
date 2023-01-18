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

class InitialInformation():
    def __init__(self,path='../',deguassian=0.03,T=300):
        self.path = path
        self.deguassian = deguassian
        self.T = T
        self.bvec = read_lattice('b', path)
        self.acvmat = read_Acv(path=path)
        self.gkkmat = read_gkk(path=path)
        self.kmap = read_kmap(path=path)
        self.kmap_dic = construct_kmap(path=path)
        self.bandmap_occ = read_bandmap(path=path)
        self.exciton_energy = read_Acv_exciton_energy(path=path)
        self.omega_mat = read_omega(path=path)  # dimension [meV]
        self.h_bar = 6.582119569E-16  # dimension = [eV.s]
        self.Constant = -1 * (2 * np.pi / self.h_bar) * (1 / int(self.kmap.shape[0]))
        f = h5.File(path+'gqQ.h5', 'r')
        self.gqQ_mat = f['data'][()]
        f.close()
        self.v = self.gqQ_mat.shape[4]
        self.m = self.gqQ_mat.shape[3]
        self.n = self.gqQ_mat.shape[2]
        self.q = self.gqQ_mat.shape[1]
        self.Q = self.gqQ_mat.shape[0]

        print("Initialized information has been created")
    def get_E_vq(self):
        """
        q = q_kmap; v =gkk_map
        :return: E_vq.shape = (v,q)
        """
        omega_vq = self.omega_mat.T
        q_gkk_index_list = self.kmap[:, 5]
        E_vq = omega_vq[:self.v, list(map(int, list(q_gkk_index_list)))]
        return E_vq * 10 ** (-3)

    def get_E_nQ(self):
        """
        Q = Q_kmap; n = acv_map
        :return: E_nQ.shape = (n,Q)
        """
        Omega_nQ = self.exciton_energy.T
        Q_acv_index_list = self.kmap[:, 3]
        E_nQ = Omega_nQ[:self.n, list(map(int, list(Q_acv_index_list)))]
        return E_nQ

    def get_E_mQq(self):
        """
        :return:E_mQq.shape = (m,Q,q)
        """
        Omega_nQ = self.exciton_energy.T
        E_mQq = np.zeros((1, self.m, 1, self.kmap.shape[0], self.kmap.shape[0]))
        for i_Q_kmap in range(self.kmap.shape[0]):
            for i_q_kmap in range(self.kmap.shape[0]):
                E_mQq[0, :, 0, i_Q_kmap, i_q_kmap] = Omega_nQ[:self.m, self.Qplusq_2_QPrimeIndexAcv(i_Q_kmap, i_q_kmap)]
        return E_mQq[0, :, 0, :, :]

    def Qplusq_2_QPrimeIndexAcv(self, Q_kmap, q_kmap):
        """
        :param Q_kmap: index of Q from 0-nQ in kmap
        :param q_kmap: index of q from 0-nq in kmap
        :return: Qpr_as_Q_plus_q_acv_index
        """
        Q_plus_q_point = move_k_back_to_BZ_1(self.kmap[Q_kmap, 0:3] + self.kmap[q_kmap, 0:3])
        key_temp = '  %.5f    %.5f    %.5f' % (Q_plus_q_point[0], Q_plus_q_point[1], Q_plus_q_point[2])
        Q_plus_q_kmapout = self.kmap_dic[key_temp.replace('-', '')]
        Qpr_as_Q_plus_q_acv_index = Q_plus_q_kmapout[0]
        return int(Qpr_as_Q_plus_q_acv_index)

    def Construct_Delta(self):
        # E.shape = (n,m,v,Q,q)
        E_nQ = self.get_E_nQ()[:, np.newaxis, np.newaxis, :, np.newaxis]
        E_mQq = self.get_E_mQq()[np.newaxis, :, np.newaxis, :, :]
        E_vq = self.get_E_vq()[np.newaxis, np.newaxis, :, np.newaxis, :]

        Delta_pos_nmvQq = Dirac_1(x=E_nQ - E_mQq + E_vq, sigma=self.deguassian)
        Delta_neg_nmvQq = Dirac_1(x=E_nQ - E_mQq - E_vq, sigma=self.deguassian)
        return Delta_pos_nmvQq, Delta_neg_nmvQq


class Solver_of_only_Q_space(InitialInformation):
    def __init__(self,path='../',degaussian=0.03,T=500,initial_occupation=0.5, nT=3000, T_total=3000, play_interval = 50):
        super(Solver_of_only_Q_space,self).__init__(path,degaussian,T)
        # Initialize F_nQ and N_vq
        # E_nQ.shape = (n,Q)
        self.F_nQ = BE(omega=self.get_E_nQ(), T=self.T)
        self.F_nQ[2,0] = initial_occupation
        self.N_vq = BE(omega=self.get_E_vq(),T=T)
        self.N_vq[0:3,0] = np.array([0,0,0])
        self.Delta_positive, self.Delta_negative = self.Construct_Delta()

        # Res Container:
        self.F_nQ_res = np.zeros((self.n, self.Q, nT))
        self.exciton_number = np.zeros(nT)
        self.dfdt_sum_res = np.zeros(nT)
        self.damping_term = np.zeros((self.n, self.Q))


        # Plot Setting:
        self.play_interval = play_interval
        self.nT = nT
        self.T_total = T_total
        self.delta_T = T_total/nT
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
        dFdt = -1 * (np.einsum('pqnmv,nmvpq,npqv->np', self.gqQ_mat, self.Delta_positive, F_abs) + np.einsum(
            'pqnmv,nmvpq,npqv->np', self.gqQ_mat, self.Delta_negative, F_em))
        return dFdt

    def solve_it(self):
        progress = ProgressBar(self.nT, fmt=ProgressBar.FULL)
        for it in range(self.nT):
            self.damping_term[:, 0] = self.F_nQ[:, 0]
            progress.current += 1
            progress()
            self.F_nQ_res[:, :, it] = self.F_nQ
            dfdt = self.__rhs_Fermi_Goldenrule(self.F_nQ)
            error_from_nosymm = dfdt.sum() / (dfdt.shape[0] * dfdt.shape[1])

            self.F_nQ = self.F_nQ + (dfdt - error_from_nosymm) * self.delta_T - self.damping_term * self.delta_T * 0.1

            # some debugging
            self.exciton_number[it] = self.F_nQ.sum()
            self.dfdt_sum_res[it] = dfdt.sum()
            # print(dfdt[2,0])

    def plot(self):
        print('plot')

        def animate(i):
            plt.clf()
            plt.scatter(self.Q_exciton, self.energy, s=self.F_nQ_res[:, :, i] * 50000)
            plt.title(label='t=%s fs' % self.Time_series[i])
            plt.xlabel("Q")
            plt.ylabel("Energy")
            plt.ylim([2, 3])
            plt.show()
            print("plotting frame %s"%i)

        fig = plt.figure()
        # animate(3)
        ani = animation.FuncAnimation(fig, animate, np.arange(self.T_total // self.play_interval), interval=10)
        # print('plot done')
        # ani.save('test.htm')

if __name__ == "__main__":
    a = Solver_of_only_Q_space(nT=300,T_total=300,play_interval=10)
    a.solve_it()
    # a.plot()

    # plt.plot(np.linspace(1,T_total,nT),F_nQ_res[1,0,:])
    # plt.plot(np.linspace(1,T_total,nT),exciton_number)