import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

# f = h5.File('EX_diffusion_evolution.h5','r')

class DiffusionFile():
    def __init__(self, file='EX_diffusion_evolution.h5') -> None:
        self.f = h5.File(file,'r')
    
    def info(self):
        print('keys of file:', self.f.keys())
        print('data dimension:', self.f['data'])

    def select_SQ_data(self,S=0, Q=0):
        return self.f['data'][S,Q]
    
class r_average(DiffusionFile):
    def __init__(self, S=0, Q=0, file='EX_diffusion_evolution.h5',
                 X=200, Y=200,delta_X=4,delta_Y=4,T=100, delta_T=0.2):
        super(r_average, self).__init__(file)
        self.S = S
        self.Q = Q
        self.data = self.select_SQ_data(S, Q)
        self.normalized_data = (self.data / np.sum(self.data,axis=(0,1))[np.newaxis, np.newaxis, :])
        self.normalized_data_X = np.sum(self.normalized_data, axis=1)
        self.normalized_data_Y = np.sum(self.normalized_data, axis=0)

        self.X = X
        self.Y = Y
        self.nX = int(X / delta_X)
        self.nY = int(Y / delta_Y)
        self.T = T
        self.nT = int(T / delta_T)
        self.time_array = np.linspace(0,self.T, self.nT)

        self.X_line = np.linspace(-self.X//2, self.X//2, self.nX) # in [A]
        self.Y_line = np.linspace(-self.Y//2, self.Y//2, self.nY) # in [A]
        self.XX, self.YY = np.meshgrid(self.X_line, self.Y_line)
        self.r_xy = np.sqrt(self.XX**2 + self.YY**2)
        self.r_x = np.abs(self.X_line)
        self.r_y = np.abs(self.Y_line)

        print('r_xy.shape:',self.r_xy.shape)
        print('r_x.shape:',self.r_x.shape)
        print('r_y.shape:',self.r_y.shape)
        print('normalized_data.shape',self.normalized_data.shape)
        print('normalized_data_x.shape',self.normalized_data_X.shape)
        print('normalized_data_y.shape',self.normalized_data_Y.shape)


        self.r_wrt_t = np.einsum('ij,ijk->k',self.r_xy, self.normalized_data)
        self.x_wrt_t = np.einsum('i,ik->k',self.r_x, self.normalized_data_X)
        self.y_wrt_t = np.einsum('i,ik->k',self.r_x, self.normalized_data_Y)    



    def plot_r(self):
        # time_array = np.linspace(0,self.T, self.nT)
        print('start plotting (getting rid of the first value)')
        x = np.vstack([self.time_array, np.ones(len(self.time_array))]).T
        self.mr, self.cr = np.linalg.lstsq(x[1:], self.r_wrt_t[1:], rcond=None)[0]
        self.mx, self.cx = np.linalg.lstsq(x[1:], self.x_wrt_t[1:], rcond=None)[0]
        self.my, self.cy = np.linalg.lstsq(x[1:], self.y_wrt_t[1:], rcond=None)[0]

        plt.plot(self.time_array[1:], self.time_array[1:] * self.mr + self.cr, color='black',linestyle='--')
        plt.plot(self.time_array[1:], self.time_array[1:] * self.mx + self.cx, color='black',linestyle='--')
        plt.plot(self.time_array[1:], self.time_array[1:] * self.my + self.cy, color='black',linestyle='--')

        plt.scatter(self.time_array, self.r_wrt_t, label='<r>(t), v=%.2f[A/fs]'%self.mr, s=1)
        plt.scatter(self.time_array, self.x_wrt_t, label='<x>(t), v=%.2f[A/fs]'%self.mx, s=1)
        plt.scatter(self.time_array, self.y_wrt_t, label='<y>(t), v=%.2f[A/fs]'%self.my, s=1)

        plt.xlabel('time [fs]]')
        plt.ylabel('distance [A]')
        plt.legend()
        plt.show()
        plt.savefig('r_average.png')
    


if __name__ == '__main__':
    r_SQ = r_average(S=0, Q=200, file='EX_diffusion_evolution.h5',
                 X=200, Y=200,delta_X=4,delta_Y=4,T=100, delta_T=0.2)
    r_SQ.plot_r()


