import numpy as np
from ELPH.EX_PH_Boltzman_Class import InitialInformation
from Common.distribution import Dirac_1, BE
from Common.progress import ProgressBar
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time
import h5py as h5
from PLot_.plot_evolution import plot_diff_evolution
import cupy as cp
from PLot_.plot_frame_evolution import plot_frame_diffusion

def Gaussian(x,y,sigma=1,x0=10,y0=10):
    return 1/(2*np.pi*sigma**2) * np.exp(-((x-x0)**2+(y-y0)**2)/(2*sigma**2))


class Solver_of_phase_space_GPU(InitialInformation):
    def __init__(self,degaussian,T,nX,nY, X,Y, delta_T,T_total,path='../', initial_S=2,initial_Q=0,initial_Gaussian_Braod=1,onGPU=True):
        super(Solver_of_phase_space_GPU,self).__init__(path=path,deguassian=degaussian,T=T,initial_S=initial_S,initial_Q=initial_Q,initial_Gaussian_Braod=initial_Gaussian_Braod,onGPU=onGPU)
        self.nX = nX
        self.nY = nY
        self.delta_T = delta_T
        self.T_total = T_total
        # self.delta_T = T_total/nT
        self.nT = int(T_total/delta_T)
        self.delta_X = X/nX
        self.delta_Y = Y/nY
        # # differential_mat = -2*np.eye(nX) + np.eye(nX,k=-1) + np.eye(nX,k=1)
        # differential_mat = -1*np.eye(nX) + np.eye(nX,k=-1)
        # self.differential_mat = differential_mat[np.newaxis,np.newaxis,:,:]

        # Initialized occupation f(n,Q,X,Y)
        self.N_vq = BE(omega=self.get_E_vq(), T=T)
        self.N_vq[0:3, 0] = np.array([0, 0, 0])
        self.Delta_positive, self.Delta_negative = self.Construct_Delta()
        # self.Delta_positive, self.Delta_negative = np.ones_like(self.Delta_positive), np.ones_like(self.Delta_negative) # TODO: open delta function later (we ignore energy conservation here)

        # Finding Group Velocity for each exciton state

        self.V_x, self.V_y = self.get_group_velocity()
        self.V_x, self.V_y = self.V_x[:,:,np.newaxis,np.newaxis]*0.02, self.V_y[:,:,np.newaxis,np.newaxis]*0.02

        # leave this for test
        # self.V_x = np.ones((self.n, self.Q))[:,:,np.newaxis,np.newaxis] * (-0.02)  #
        # self.V_y = np.ones((self.n, self.Q))[:,:,np.newaxis,np.newaxis] * 0.03
        # self.V_x[0,0,:,:] = np.ones((1,1))[:,:,np.newaxis,np.newaxis] * 0.  #
        # self.V_y[0,0,:,:] = np.ones((1,1))[:,:,np.newaxis,np.newaxis] * 0.
        # self.V_x[0,2,:,:] = np.ones((1,1))[:,:,np.newaxis,np.newaxis] * -0.035 #
        # self.V_y[0,2,:,:] = np.ones((1,1))[:,:,np.newaxis,np.newaxis] * (-0.0)
        # self.V_x[0,3,:,:] = np.ones((1,1))[:,:,np.newaxis,np.newaxis] * 0.025  #
        # self.V_y[0,3,:,:] = np.ones((1,1))[:,:,np.newaxis,np.newaxis] * (0.)


        # Lax-wendroff:
        C = self.V_x * self.delta_T / self.delta_X
        a1 = -1 * C * (1 - C) / 2
        a_neg1 = C * (1 + C) / 2
        a0 = -C ** 2

        self.differential_mat = np.eye(nX, k=-1) * a_neg1 + np.eye(nX) * a0 + np.eye(nX, k=1) * a1
        self.differential_mat[:,:, -1, 0] =  a1[:,:, 0, 0]
        self.differential_mat[:,:, 0, -1] = a_neg1[:,:, 0, 0]

        C_y = self.V_y * self.delta_T / self.delta_Y
        a1 = -1 * C_y * (1 - C_y) / 2
        a_neg1 = C_y * (1 + C_y) / 2
        a0 = -C_y ** 2

        self.differential_mat_y = np.eye(nY, k=-1) * a_neg1 + np.eye(nY) * a0 + np.eye(nY, k=1) * a1
        self.differential_mat_y[:,:, -1, 0] = a1[:,:, 0, 0]
        self.differential_mat_y[:,:, 0, -1] =  a_neg1[:,:, 0, 0]

        # Lax-wendroff<


        # TODOdone: find a way to initialize this
        self.ini_x = np.arange(0, X, self.delta_X)
        self.ini_y = np.arange(0, Y, self.delta_Y)
        self.ini_xx, self.ini_yy = np.meshgrid(self.ini_x, self.ini_y)

        self.F_nQxy = np.zeros((self.n,self.Q,self.nX,self.nY))
        self.F_nQxy[self.initial_S,self.initial_Q,:,:] = Gaussian(self.ini_xx,self.ini_yy,x0=X//2,y0=Y//2,sigma=self.initial_Gaussian) # tododone: add parameter to specify exciton state you want to initialize

        self.F_nQxy_res = np.zeros((self.n, self.Q,self.nX,self.nY, self.nT))

        self.damping_term = np.zeros((self.n, self.Q, self.nX ,self.nY))
        self.dfdt_res = np.zeros((self.n, self.Q,self.nX,self.nY, self.nT))


        self.Qq_2_Qpr_res = self.Qq_2_Qpr_kmap()
        ### Debug:
        # self.V_x = np.ones((1,1))[:,:,np.newaxis,np.newaxis] * (0.4)   #TODOdone: use Omega(S,Q)
        # self.V_y = np.ones((1, 1))[:,:,np.newaxis,np.newaxis] * 0.4
        # self.F_nQxy = np.ones((1,1,self.nX,self.nY))
        # self.F_nQxy[0,0,:,:] = Gaussian(ini_xx,ini_yy) * 20
        # self.F_nQxy_res = np.zeros((1,1,self.nX,self.nY, self.nT))
        # self.damping_term = np.zeros((1,1, self.nX ,self.nY))

        print("Initialized information has been loaded")


    def kmap_check_for_derivative(self):
        return True # TODO

    def get_group_velocity(self):
        if not self.kmap_check_for_derivative(): raise Exception("group velocity can not be calculated with this kmap, please make sure that you are using same size Q, k_acv, q and k_gkk in previous calculation")
        E_nQ = self.get_E_nQ()
        nQ_sqrt = int(np.sqrt(self.Q))
        nS = int(self.n)
        b1, b2 = self.bvec[0, :2], self.bvec[1, :2]
        cos_theta_b1, sin_theta_b1 = b1[0] / np.linalg.norm(b1), b1[1] / np.linalg.norm(b1)
        cos_theta_b2, sin_theta_b2 = b2[0] / np.linalg.norm(b1), b2[1] / np.linalg.norm(b1)
        delta_Qb1 = (self.kmap[nQ_sqrt, 0] - self.kmap[0, 0]) * np.linalg.norm(b1)  # todo: this can be used for kmap check!
        delta_Qb2 = (self.kmap[1, 1] - self.kmap[0, 1]) * np.linalg.norm(b2)  # angstrom-1
        diff_mat_central = np.eye(nQ_sqrt, k=-1) + -1 * np.eye(nQ_sqrt, k=1)
        diff_mat_central[0, -1] = 1
        diff_mat_central[-1, 0] = -1
        diff_mat_central_b1 = (diff_mat_central / (delta_Qb1 * self.h_bar))[np.newaxis, :, :]
        diff_mat_central_b2 = (diff_mat_central / (delta_Qb2 * self.h_bar))[np.newaxis, :, :]

        V_b1 = np.matmul(diff_mat_central_b1, E_nQ.reshape((self.n, nQ_sqrt, nQ_sqrt)))
        V_b2 = np.matmul(E_nQ.reshape((self.n, nQ_sqrt, nQ_sqrt)), diff_mat_central_b2)

        V_nQQ_x = V_b1 * cos_theta_b1 + V_b2 * cos_theta_b2
        V_nQQ_y = V_b1 * sin_theta_b1 + V_b2 * sin_theta_b2
        return V_nQQ_x.reshape((self.n,int(nQ_sqrt**2))), V_nQQ_y.reshape((self.n,int(nQ_sqrt**2)))

    def Qq_2_Qpr_kmap(self): #TODO: it should be in parent class
        """
        start from 0
        """
        Q_acv_back2_kmap = list(map(int, list(self.kmap[:, 3])))
        Qq_2_Qpr_res = np.zeros((self.Q,self.q),dtype=int)
        for i_Q_kmap in range(self.Q):
            for i_q_kmap in range(self.q):
                Qq_2_Qpr_res[i_Q_kmap, i_q_kmap] = Q_acv_back2_kmap.index(self.Qplusq_2_QPrimeIndexAcv(i_Q_kmap, i_q_kmap))
        return Qq_2_Qpr_res

    def __update_F_nQqxy(self,F_nQxy_last):
        """
        :param F_nQ:
        :return: F_nQq.shape = (n,Q,q)
        """
        # Q_acv_back2_kmap = list(map(int, list(self.kmap[:, 3])))
        # # self.F_nQxy.shape = (n,Q,q,nX,nY)
        # F_mQqxy = np.zeros((F_nQxy_last.shape[0], F_nQxy_last.shape[1], F_nQxy_last.shape[1],F_nQxy_last.shape[2],F_nQxy_last.shape[3]))
        # for i_Q_kmap in range(self.Q):
        #     for i_q_kmap in range(self.q):
        #         F_mQqxy[:, i_Q_kmap, i_q_kmap,:,:] = F_nQxy_last[:,
        #                                        Q_acv_back2_kmap.index(self.Qplusq_2_QPrimeIndexAcv(i_Q_kmap, i_q_kmap)),:,:]
        # return F_mQqxy
        return F_nQxy_last[:,self.Qq_2_Qpr_res,:,:]


    def __rhs_Fermi_Goldenrule_GPU(self,F_nQxy_last):
        # dFdt = (n,Q,x,y)
        # TODO: Optimization
        t0 = time.time()
        t1 = time.time()
        F_mQqxy = self.__update_F_nQqxy(F_nQxy_last)
        t2_update =time.time()
        # print('  (1) time for updating F_mQq', time.time() - t0, 's')

        # load data from CPU to GPU memory: >>>>>>>>>>
        t0 = time.time()
        F_mQqxy = cp.array(F_mQqxy)
        self.N_vq = cp.array(self.N_vq)
        self.gqQ_mat = cp.array(self.gqQ_mat)
        self.Delta_positive = cp.array(self.Delta_positive)
        self.Delta_negative = cp.array(self.Delta_negative)
        F_nQxy_last = cp.array(F_nQxy_last)
        # print('  (2) time for loading data from cpu to gpu', time.time() - t0, 's')
        # -----------------------------------<<<<<<<<<<

        t0 = time.time()
        F_abs = cp.einsum('npxy,vq,mpqxy->nmpqvxy',F_nQxy_last, self.N_vq, 1 + F_mQqxy,optimize='optimal') \
                - cp.einsum('npxy,vq,mpqxy->nmpqvxy', 1 + F_nQxy_last, 1 + self.N_vq, F_mQqxy,optimize='optimal')
        F_em = cp.einsum('npxy,vq,mpqxy->nmpqvxy', F_nQxy_last, 1 + self.N_vq, 1 + F_mQqxy,optimize='optimal') \
                - cp.einsum('npxy,vq,npqxy->npqvxy', 1 + F_nQxy_last, self.N_vq, F_mQqxy,optimize='optimal')
        # Debugging: 02/11/2023 n --> m  !!!! Bowen Hou
        dFdt =  cp.einsum('pqnmv,nmvpq,nmpqvxy->npxy', self.gqQ_mat, self.Delta_positive, F_abs,optimize='optimal') \
                + cp.einsum('pqnmv,nmvpq,nmpqvxy->npxy', self.gqQ_mat, self.Delta_negative, F_em,optimize='optimal')

        # print('  (3) time for GPU calculation', time.time() - t0, 's')
        t2 = time.time()

        t0 = time.time()
        dFdt = cp.asnumpy(dFdt)
        # print('  (4) time for transfer data to CPU', time.time() - t0, 's')

        return -1*(np.pi * 2)/(self.h_bar * self.Q) * dFdt, t2-t2_update, t2_update - t1

    def __rhs_Fermi_Goldenrule_CPU(self,F_nQxy_last):
        t1 = time.time()
        F_mQqxy = self.__update_F_nQqxy(F_nQxy_last)
        t2_update = time.time()

        F_abs = np.einsum('npxy,vq,mpqxy->nmpqvxy',F_nQxy_last, self.N_vq, 1 + F_mQqxy,optimize='optimal') \
                - np.einsum('npxy,vq,mpqxy->nmpqvxy', 1 + F_nQxy_last, 1 + self.N_vq, F_mQqxy,optimize='optimal')
        F_em = np.einsum('npxy,vq,mpqxy->nmpqvxy', F_nQxy_last, 1 + self.N_vq, 1 + F_mQqxy,optimize='optimal') \
                - np.einsum('npxy,vq,npqxy->npqvxy', 1 + F_nQxy_last, self.N_vq, F_mQqxy,optimize='optimal')
        # Debugging: 02/11/2023 n --> m  !!!! Bowen Hou
        dFdt =  np.einsum('pqnmv,nmvpq,nmpqvxy->npxy', self.gqQ_mat, self.Delta_positive, F_abs,optimize='optimal') \
                + np.einsum('pqnmv,nmvpq,nmpqvxy->npxy', self.gqQ_mat, self.Delta_negative, F_em,optimize='optimal')

        t2 = time.time()
        return -1*(np.pi * 2)/(self.h_bar * self.Q) * dFdt, t2-t2_update, t2_update - t1


    def solve_it(self):
        if self.onGPU:
            print('[GPU acceleration] ON')
        else:
            print('[GPU acceleration] OFF')

        progress = ProgressBar(self.nT, fmt=ProgressBar.FULL)
        time_rhs = 0
        time_rhs_update_F_nQ = 0
        time_total_start = time.time()
        for it in range(self.nT):
            t0 = time.time()
            self.damping_term[0, 0, :, :] = self.F_nQxy[0, 0, :, :]
            progress.current += 1
            progress()
            self.F_nQxy_res[:, :, :,:,it] = self.F_nQxy

            if self.onGPU:
                dfdt, time_rhs_Fermi_temp, time_rhs_Fermi_update_F_nQ_temp = self.__rhs_Fermi_Goldenrule_GPU(self.F_nQxy)
            else:
                dfdt, time_rhs_Fermi_temp, time_rhs_Fermi_update_F_nQ_temp = self.__rhs_Fermi_Goldenrule_CPU(self.F_nQxy)

            # print('\ntime for F_nQ (%s / %s):' % (it, self.nT), time.time() - t0, 's')
            self.dfdt_res[:,:,:,:,it] = dfdt

            time_rhs += time_rhs_Fermi_temp
            time_rhs_update_F_nQ += time_rhs_Fermi_update_F_nQ_temp
            # error_from_nosymm = dfdt.sum() / (dfdt.shape[0] * dfdt.shape[1] * dfdt.shape[2] * dfdt.shape[3])

            self.F_nQxy = self.F_nQxy  \
                          + (dfdt) * self.delta_T\
                          - ( np.matmul(self.F_nQxy, self.differential_mat)
                              + np.matmul(self.differential_mat_y, self.F_nQxy)) \
                          # - self.damping_term * self.delta_T * 0


        time_total_end= time.time()

        print('\ntotal time to solve this equation:',time_total_end - time_total_start)
        print('total time for GPU matrix:',time_rhs)
        print('total time for updating F_mQq:', time_rhs_update_F_nQ)
            # self.F_nQxy = self.F_nQxy - self.damping_term * self.delta_T * 0.0+ ( np.matmul(self.F_nQxy, self.differential_mat) * self.V_x / self.delta_X
            #                   + np.matmul(self.differential_mat, self.F_nQxy) *  self.V_y  /self.delta_Y ) * self.delta_T
    def write_diffusion_evolution(self):
        f = h5.File(self.path+'EX_diffusion_evolution.h5','w')
        f.create_dataset('data',data=self.F_nQxy_res)
        f.close()
        print('EX_diffusion_evolution.h5 has been written')

    def plot(self,n_plot,play_interval=2,saveformat=None,readfromh5=False):
        """
        :param n_plot: state you want to see: start from 1,2,3...
        :param play_inverval:  plot evolution in every "plat_interval" [fs]
        :return:
        """
        if readfromh5:
            f = h5.File(self.path+'EX_diffusion_evolution.h5','r')
            self.F_nQxy_res = f['data'][()]
            f.close()

        n_plot = n_plot
        ani = plot_diff_evolution(F_nQxy_res=self.F_nQxy_res,
                                  nX=self.nX,
                                  nY=self.nY,
                                  n=n_plot,
                                  T_total=self.T_total,
                                  delta_T=self.delta_T,
                                  play_interval=play_interval,
                                  path=self.path,
                                  saveformat=saveformat
                                  )
        return ani




if __name__ == "__main__":
############## Solve PDF and plot

    a = Solver_of_phase_space_GPU(degaussian=0.05,delta_T=1, T_total=200,T=100,nX=80,nY=80, X=20,Y=20,path='../',initial_S=2,initial_Q=0,initial_Gaussian_Braod=1,onGPU=True)
    a.solve_it()
    a.write_diffusion_evolution()
    ani = a.plot(n_plot=2,play_interval=1,saveformat=None,readfromh5=True)


    # plot_frame_diffusion

