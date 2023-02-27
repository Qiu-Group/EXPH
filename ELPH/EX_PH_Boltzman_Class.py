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
    def __init__(self,path='../',deguassian=0.03,T=300,initial_S=2,initial_Q=0,initial_Gaussian_Braod=1, high_symm="0.0 0.0 0.0, 0.75 0.75 0, 0.75 0.0 0.0, 0.0 0.0 0.0", onGPU=True):
        self.path = path
        self.deguassian = deguassian
        self.T = T
        self.bvec = read_lattice('b', path)

        # self.acvmat = read_Acv(path=path)
        # self.gkkmat = read_gkk(path=path)

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

        self.initial_Q = initial_Q
        self.initial_S = initial_S
        self.initial_Gaussian = initial_Gaussian_Braod
        self.onGPU = onGPU
        self.high_symm = high_symm

        #print("Initialized information has been created")
    def get_E_vq(self):
        """
        q = q_kmap; v =gkk_map
        :return: E_vq.shape = (v,q)
        """
        omega_vq = self.omega_mat.T
        q_gkk_index_list = self.kmap[:, 5]
        E_vq = omega_vq[:self.v, list(map(int, list(q_gkk_index_list)))]
        return E_vq * 10 ** (-3) # meV --> eV

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

        # return Dirac_1(x=E_nQ - E_mQq + E_vq, sigma=self.deguassian), Dirac_1(x=E_nQ - E_mQq - E_vq, sigma=self.deguassian)

        Delta_pos_nmvQq = Dirac_1(x=E_nQ - E_mQq + E_vq, sigma=self.deguassian)
        Delta_neg_nmvQq = Dirac_1(x=E_nQ - E_mQq - E_vq, sigma=self.deguassian)
        return Delta_pos_nmvQq, Delta_neg_nmvQq

    def get_highsymmetry_index_kmap(self):
        """
        :return: [index1, index2, ...] which is index for high symmetry k-points and lines
        """
        print('------------finding high symmetry path------------')
        tolerence = 1E-5
        kmap_points = self.kmap[:,:2]

        # (i) get the index of high symmetry point you set
        high_symm_list = self.high_symm.split(',')
        high_symm_kpoints_kmap_index = [] # [high_symm1, high_symm2, ...]
        for h_kpt in high_symm_list:
            found_high = False
            h_kpt_array = np.fromstring(h_kpt.strip(),dtype=float,sep=' ')[:2]
            True_False_matrix_to_be_determined = (np.abs(kmap_points - h_kpt_array) < tolerence)
            for i in range(kmap_points.shape[0]):
                if np.all(True_False_matrix_to_be_determined[i]):
                    high_symm_kpoints_kmap_index.append([i,i//int(np.sqrt(self.Q)),i%int(np.sqrt(self.Q))])
                    print('index of point: %.5f %.f5 0.00000 '%(h_kpt_array[0], h_kpt_array[1]),' in kmap :',i,' [%s,%s]'%(i//int(np.sqrt(self.Q)),i%int(np.sqrt(self.Q))))
                    found_high =True
                    break
            if not found_high:
                raise Exception('kpoint: '+h_kpt+' can not be found in kkqQmap.dat')
        # print(high_symm_kpoints_kmap_index)

        # (ii) get the index list of high symmetry line

        res_matrix_index = []
        res_kmap_index = []
        for i in range(len(high_symm_kpoints_kmap_index)-1):
            # print('\n-------')
            temp_start_point = np.array(high_symm_kpoints_kmap_index[i][1:])
            temp_end_point = np.array(high_symm_kpoints_kmap_index[i+1][1:])


            x_range = np.linspace(temp_start_point[0], temp_end_point[0] + 1 if temp_start_point[0] > temp_end_point[0] else temp_end_point[0] - 1, np.abs(temp_end_point[0] - temp_start_point[0]))
            y_range = np.linspace(temp_start_point[1], temp_end_point[1] + 1 if temp_start_point[1] > temp_end_point[1] else temp_end_point[1] - 1 , np.abs(temp_end_point[1] - temp_start_point[1]))

            # print(x_range, y_range)

            if len(x_range) == 0:
                x_range = np.ones_like(y_range) * temp_start_point[0]
            if len(y_range) == 0:
                y_range = np.ones_like(x_range) * temp_start_point[1]


            if max(len(y_range),len(x_range)) % min(len(x_range),len(y_range)) == 0:
                single_period = max(len(y_range), len(x_range)) // min(len(x_range), len(y_range))
                if len(y_range) > len(x_range):
                    y_range = np.linspace(temp_start_point[1], temp_end_point[1] + single_period if temp_start_point[1] > temp_end_point[1] else temp_end_point[1] - single_period , len(x_range))
                elif len(x_range) > len(y_range):
                    x_range = np.linspace(temp_start_point[0], temp_end_point[0] + single_period if temp_start_point[0] > temp_end_point[0] else temp_end_point[0] - single_period, len(y_range))

                # print(x_range,y_range)
                for j in range(len(x_range)): #todo: add index
                    res_matrix_index.append([int(x_range[j]), int(y_range[j])])
                    res_kmap_index.append( int((x_range[j]) * np.sqrt(self.Q) + y_range[j] ) )
                    # print([int(x_range[j]), int(y_range[j])])
            else:
                res_matrix_index.append([int(x_range[0]), int(y_range[0])])
                res_kmap_index.append(int((x_range[0]) * np.sqrt(self.Q) + y_range[0] ) )
                # print([int(x_range[0]), int(y_range[0])])
        # add the last point
        # res_matrix_index.append(high_symm_kpoints_kmap_index[-1][1:])
        # res_kmap_index.append(int(high_symm_kpoints_kmap_index[-1][1]*np.sqrt(self.Q) + high_symm_kpoints_kmap_index[-1][2]))
        print('high symmetry path includes %s k points'%len(res_kmap_index))
        # print("matrix path:",res_matrix_index)
        print("kmap path",res_kmap_index)
        print('------------finding high symmetry path done------------')
        return res_kmap_index


if __name__ == "__main__":
    a = InitialInformation(high_symm="0.75 0.75 0.0, 0.0 0.75 0.0, 0.25 0.25 0.0")
    a.get_highsymmetry_index_kmap()