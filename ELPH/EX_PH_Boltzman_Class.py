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
        self.h_bar = 6.582119569E-16 * 1E15  # dimension = [eV.fs]
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
        Q = Q_kmap: 0,1,2,3... line in kmap; n = acv_map
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