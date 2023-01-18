import numpy as np
from Parallel.Para_rt_Boltzmann_Class import InitialInformation
from Common.distribution import Dirac_1, BE
from Common.progress import ProgressBar
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def Gaussian(x,y,sigma=1,x0=15,y0=15):
    return 1/(2*np.pi*sigma**2) * np.exp(-((x-x0)**2+(y-y0)**2)/(2*sigma**2))


class Solver_of_phase_space(InitialInformation):
    def __init__(self,path='../',degaussian=0.05,T=300,nX=10,nY=10, X=10,Y=10, nT=300,T_total=300):
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
        # self.V_x = np.ones((self.n, self.Q))[:,:,np.newaxis,np.newaxis] * 0.0001   #TODO: use Omega(S,Q)
        # self.V_y = np.ones((self.n, self.Q))[:,:,np.newaxis,np.newaxis] * 0.0001

        # TODO: find a way to initialize this
        ini_x = np.arange(0, X, self.delta_X)
        ini_y = np.arange(0, Y, self.delta_Y)

        self.ini_xx, self.ini_yy = np.meshgrid(ini_x, ini_y)

        # self.F_nQxy = np.ones((self.n,self.Q,self.nX,self.nY))
        # self.F_nQxy[2,0,:,:] = Gaussian(ini_xx,ini_yy) * 10
        #
        # self.F_nQxy_res = np.zeros((self.n, self.Q,self.nX,self.nY, self.nT))
        self.damping_term = np.zeros((1,1, self.nX ,self.nY))


        ### Debug:
        self.V_x = np.ones((1,1))[:,:,np.newaxis,np.newaxis] * (0.)   #TODO: use Omega(S,Q)
        self.V_y = np.ones((1, 1))[:,:,np.newaxis,np.newaxis] * (0.4)
        self.F_nQxy = np.ones((1,1,self.nX,self.nY))
        self.F_nQxy[0,0,:,:] = Gaussian(self.ini_xx,self.ini_yy) * 20
        self.F_nQxy_res = np.zeros((1,1,self.nX,self.nY, self.nT))
        # self.damping_term = np.zeros((self.n, self.Q, self.nX ,self.nY))
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
        F_nQqxy = self.__update_F_nQqxy(F_nQxy_last)
        F_abs = np.einsum('npxy,vq,npqxy->npqvxy',F_nQxy_last, self.N_vq, 1 + F_nQqxy) - np.einsum('npxy,vq,npqxy->npqvxy', 1 + F_nQxy_last, 1 + self.N_vq,
                                                                                F_nQqxy)
        F_em = np.einsum('npxy,vq,npqxy->npqvxy', F_nQxy_last, 1 + self.N_vq, 1 + F_nQqxy) - np.einsum('npxy,vq,npqxy->npqvxy', 1 + F_nQxy_last, self.N_vq,
                                                                                   F_nQqxy)
        dFdt = -1 * (np.einsum('pqnmv,nmvpq,npqvxy->npxy', self.gqQ_mat, self.Delta_positive, F_abs) + np.einsum(
            'pqnmv,nmvpq,npqvxy->npxy', self.gqQ_mat, self.Delta_negative, F_em))
        return dFdt

    def solve_it(self):
        progress = ProgressBar(self.nT, fmt=ProgressBar.FULL)
        for it in range(self.nT):

            self.damping_term[0, 0, :, :] = self.F_nQxy[0, 0, :, :]
            progress.current += 1
            progress()
            self.F_nQxy_res[:, :, :,:,it] = self.F_nQxy

            self.F_nQxy = self.F_nQxy - self.damping_term * self.delta_T * 0.1+ ( np.matmul(self.F_nQxy, self.differential_mat) * self.V_x / self.delta_X
                              + np.matmul(self.differential_mat, self.F_nQxy) *  self.V_y  /self.delta_Y ) * self.delta_T \
                              # + Gaussian(self.ini_xx,self.ini_yy,x0=15,y0=10) * 20 *0.1

if __name__ == "__main__":
    Q=0
    n=0
    nX = 60
    nY = 60
    T_total = 400
    nT = 800
    play_interval = 5

    a = Solver_of_phase_space(degaussian=0.03,T=300,nX=nX,nY=nY, X=30,Y=30, nT=nT,T_total=T_total)
    a.solve_it()

    X = np.arange(nX)
    Y = np.arange(nY)

    XX, YY = np.meshgrid(X,Y)

    fig =plt.figure()
    Time_series = np.arange(0, T_total, play_interval)
    def animate(i):
        plt.clf()
        # plt.contourf(XX,YY,a.F_nQxy_res[n,Q,:,:,i],levels=np.linspace(0,a.F_nQxy_res[n,Q,:,:,0].max(),80))
        # plt.colorbar()
        #
        plt.plot(a.F_nQxy_res[n,Q,:,0,i])
        plt.ylim(0,a.F_nQxy_res[n,Q,:,0,0].max())

        plt.title(label='t=%s fs' % Time_series[i])





    ani = animation.FuncAnimation(fig, animate, np.arange(T_total // play_interval), interval=50)
    plt.show()
    # ani.save('test2.html')
