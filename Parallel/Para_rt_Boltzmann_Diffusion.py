import numpy as np
from Parallel.Para_rt_Boltzmann_Class import InitialInformation
from Common.distribution import Dirac_1, BE
from Common.progress import ProgressBar
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time

def Gaussian(x,y,sigma=2,x0=20,y0=20):
    return 1/(2*np.pi*sigma**2) * np.exp(-((x-x0)**2+(y-y0)**2)/(2*sigma**2))


class Solver_of_phase_space(InitialInformation):
    def __init__(self,path='../',degaussian=0.01,T=300,nX=10,nY=10, X=10,Y=10, nT=300,T_total=300):
        super(Solver_of_phase_space,self).__init__(path,degaussian,T)
        self.nX = nX
        self.nY = nY
        self.nT = nT
        self.T_total = T_total
        self.delta_T = T_total/nT
        self.delta_X = X/nX
        self.delta_Y = Y/nY
        # differential_mat = -2*np.eye(nX) + np.eye(nX,k=-1) + np.eye(nX,k=1)
        differential_mat = -1*np.eye(nX) + np.eye(nX,k=-1)
        self.differential_mat = differential_mat[np.newaxis,np.newaxis,:,:]

        # Initialized occupation f(n,Q,X,Y)
        self.N_vq = BE(omega=self.get_E_vq(), T=T)
        self.N_vq[0:3, 0] = np.array([0, 0, 0])
        self.Delta_positive, self.Delta_negative = self.Construct_Delta()
        self.V_x = np.ones((self.n, self.Q))[:,:,np.newaxis,np.newaxis] * 0.  #TODO: use Omega(S,Q)
        self.V_y = np.ones((self.n, self.Q))[:,:,np.newaxis,np.newaxis] * 0.
        self.V_x[0,1,:,:] = np.ones((1,1))[:,:,np.newaxis,np.newaxis] * 0.1   #TODO: use Omega(S,Q)
        self.V_y[0,1,:,:] = np.ones((1,1))[:,:,np.newaxis,np.newaxis] * 0.1

        # TODO: find a way to initialize this
        self.ini_x = np.arange(0, X, self.delta_X)
        self.ini_y = np.arange(0, Y, self.delta_Y)

        self.ini_xx, self.ini_yy = np.meshgrid(self.ini_x, self.ini_y)

        self.F_nQxy = np.zeros((self.n,self.Q,self.nX,self.nY))
        self.F_nQxy[0,0,:,:] = self.F_nQxy[0,0,:,:] + Gaussian(self.ini_xx,self.ini_yy) * 1

        self.F_nQxy_res = np.zeros((self.n, self.Q,self.nX,self.nY, self.nT))
        self.damping_term = np.zeros((self.n, self.Q, self.nX ,self.nY))

        self.dfdt_res = np.zeros((self.n, self.Q,self.nX,self.nY, self.nT))

        ### Debug:
        # self.V_x = np.ones((1,1))[:,:,np.newaxis,np.newaxis] * (0.4)   #TODO: use Omega(S,Q)
        # self.V_y = np.ones((1, 1))[:,:,np.newaxis,np.newaxis] * 0.4
        # self.F_nQxy = np.ones((1,1,self.nX,self.nY))
        # self.F_nQxy[0,0,:,:] = Gaussian(ini_xx,ini_yy) * 20
        # self.F_nQxy_res = np.zeros((1,1,self.nX,self.nY, self.nT))
        # self.damping_term = np.zeros((1,1, self.nX ,self.nY))

        print("Initialized information has been loaded")

    def __update_F_nQqxy(self,F_nQxy_last):
        """
        :param F_nQ:
        :return: F_nQq.shape = (n,Q,q)
        """
        Q_acv_back2_kmap = list(map(int, list(self.kmap[:, 3])))
        # self.F_nQxy.shape = (n,Q,q,nX,nY)
        F_nQqxy = np.zeros((F_nQxy_last.shape[0], F_nQxy_last.shape[1], F_nQxy_last.shape[1],F_nQxy_last.shape[2],F_nQxy_last.shape[3]))
        for i_Q_kmap in range(self.Q):
            for i_q_kmap in range(self.q):
                F_nQqxy[:, i_Q_kmap, i_q_kmap,:,:] = F_nQxy_last[:,
                                               Q_acv_back2_kmap.index(self.Qplusq_2_QPrimeIndexAcv(i_Q_kmap, i_q_kmap)),:,:]
        return F_nQqxy

    def __rhs_Fermi_Goldenrule(self,F_nQxy_last):
        # dFdt = (n,Q,x,y)
        # TODO: Optimization
        t1 = time.time()
        F_nQqxy = self.__update_F_nQqxy(F_nQxy_last)
        t2_update =time.time()
        F_abs = np.einsum('npxy,vq,npqxy->npqvxy',F_nQxy_last, self.N_vq, 1 + F_nQqxy,optimize='optimal') - np.einsum('npxy,vq,npqxy->npqvxy', 1 + F_nQxy_last, 1 + self.N_vq,
                                                                                F_nQqxy,optimize='optimal')
        F_em = np.einsum('npxy,vq,npqxy->npqvxy', F_nQxy_last, 1 + self.N_vq, 1 + F_nQqxy,optimize='optimal') - np.einsum('npxy,vq,npqxy->npqvxy', 1 + F_nQxy_last, self.N_vq,
                                                                                   F_nQqxy,optimize='optimal')
        dFdt = -1E70 * (np.einsum('pqnmv,nmvpq,npqvxy->npxy', self.gqQ_mat, self.Delta_positive, F_abs,optimize='optimal') + np.einsum(
            'pqnmv,nmvpq,npqvxy->npxy', self.gqQ_mat, self.Delta_negative, F_em,optimize='optimal'))
        t2 = time.time()
        return dFdt, t2-t1, t2_update - t1

    def solve_it(self):
        progress = ProgressBar(self.nT, fmt=ProgressBar.FULL)
        time_rhs = 0
        time_rhs_update_F_nQ = 0
        time_total_start = time.time()
        for it in range(self.nT):

            self.damping_term[0, 0, :, :] = self.F_nQxy[0, 0, :, :]
            progress.current += 1
            progress()
            self.F_nQxy_res[:, :, :,:,it] = self.F_nQxy
            dfdt,time_rhs_Fermi_temp,time_rhs_Fermi_update_F_nQ_temp = self.__rhs_Fermi_Goldenrule(self.F_nQxy)

            self.dfdt_res[:,:,:,:,it] = dfdt

            time_rhs += time_rhs_Fermi_temp
            time_rhs_update_F_nQ += time_rhs_Fermi_update_F_nQ_temp
            error_from_nosymm = dfdt.sum() / (dfdt.shape[0] * dfdt.shape[1] * dfdt.shape[2] * dfdt.shape[3])

            self.F_nQxy = self.F_nQxy  - self.damping_term * self.delta_T * 0. + (dfdt - error_from_nosymm) * self.delta_T\
                          - ( np.matmul(self.F_nQxy, self.differential_mat) * self.V_x / self.delta_X
                              + np.matmul(self.differential_mat, self.F_nQxy) *  self.V_y  /self.delta_Y ) * self.delta_T


        time_total_end= time.time()

        print('\ntotal time to solve this equation:',time_total_end - time_total_start)
        print('total time to solve rhs_Fermi:',time_rhs)
        print('total time to solve rhs_Fnq_Fermi:', time_rhs_update_F_nQ)
            # self.F_nQxy = self.F_nQxy - self.damping_term * self.delta_T * 0.0+ ( np.matmul(self.F_nQxy, self.differential_mat) * self.V_x / self.delta_X
            #                   + np.matmul(self.differential_mat, self.F_nQxy) *  self.V_y  /self.delta_Y ) * self.delta_T

if __name__ == "__main__":
    Q=1
    n=0
    nX = 120
    nY = 120
    T_total = 100
    nT = 400
    play_interval = 2

    a = Solver_of_phase_space(degaussian=0.005,T=100,nX=nX,nY=nY, X=40,Y=40, nT=nT,T_total=T_total)
    a.solve_it()

    X = np.arange(nX)
    Y = np.arange(nY)

    XX, YY = np.meshgrid(X,Y)

    fig =plt.figure()
    Time_series = np.arange(0, T_total, play_interval)
    def animate(i):
        plt.clf()
        if Q == 0:
            plt.contourf(XX,YY,a.F_nQxy_res[n,Q,:,:,i],levels=np.linspace(a.F_nQxy_res[n,Q,:,:,0].min()-0.01,a.F_nQxy_res[n,Q,:,:,0].max(),80))
        else:
            plt.contourf(XX, YY, a.F_nQxy_res[n, Q, :, :, i],
                         levels=np.linspace(0,a.F_nQxy_res[n,Q,:,:,:].max(),80))
        # plt.title(label='t=%s fs'%Time_series[i])
        plt.title('t=%s fs'%Time_series[i])

        plt.colorbar()



    ani = animation.FuncAnimation(fig, animate, np.arange(T_total // play_interval), interval=10)
    plt.show()
    # ani.save('test2.gif')
