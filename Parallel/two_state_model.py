import numpy as np
from Common.progress import ProgressBar
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


def Gaussian(x,y,sigma=1,x0=10,y0=10):
    return 1/(2*np.pi*sigma**2) * np.exp(-((x-x0)**2+(y-y0)**2)/(2*sigma**2))

def update_F_nQq(F_qxy_last,g_qp):
    """
    :param F_Qxy_last:
    :param g_qp:
    :return: F(q,x,y) = Sum(p){g_qp * F(q,x,y) * (F(p,x,y) + 1)} <- Bosons
    """
    return -0.1 * (np.einsum('qp,qxy,pxy -> qxy',g_qp,F_qxy_last,1+F_qxy_last) - np.einsum('pq,pxy,qxy -> qxy',g_qp,F_qxy_last,1+F_qxy_last))

def solve_it(F_qxy_res,F_qxy, dfdt_res,diff_mat, V_x, delta_T, delta_X, delta_Y, V_y):
    progress = ProgressBar(nT, fmt=ProgressBar.FULL)
    for i in range(nT):
        progress.current += 1
        progress()

        dfdt = update_F_nQq(F_qxy_last=F_qxy, g_qp=g_qp)
        dfdt_res[:,:,:,i] = dfdt[:,:,:]
        F_qxy_res[:, :, :, i] = F_qxy[:, :, :]

        F_qxy = F_qxy + delta_T * dfdt * delta_T  \
                + (np.matmul(F_qxy, diff_mat) * V_x) * delta_T / delta_X \
                + (np.matmul(diff_mat,F_qxy ) * V_y) * delta_T / delta_Y


X, Y = 20,20
nX, nY = 40, 40
T_total = 400
nT = 800
delta_X, delta_Y = X / nX , Y / nY
delta_T = T_total / nT

ini_x = np.arange(0, X, delta_X)
ini_y = np.arange(0, Y, delta_Y)
ini_xx, ini_yy = np.meshgrid(ini_x, ini_y)

g_qp = np.array([[0,1],[0.3,0]]) # .shape = (2,2)
F_qxy = np.zeros((2,nX,nY)) # .shape = (2,nX,nY)
F_qxy[0] = Gaussian(ini_xx,ini_yy)  # Initial Ccondition
F_qxy_res = np.zeros((2,nX,nY,nT)) # .shape = (2,nX,nY,nT)
dfdt_res = np.zeros((2,nX,nY,nT))
differential_mat = -1*np.eye(nX) + np.eye(nX,k=-1)
V_x = np.array([0., 0.3])[:,np.newaxis,np.newaxis] # .shape(2,1,1)
V_y = np.array([0, 0.2])[:,np.newaxis,np.newaxis] # .shape(2,1,1)

# solve and plot
solve_it(F_qxy_res=F_qxy_res, F_qxy=F_qxy, dfdt_res=dfdt_res, delta_T=delta_T, delta_X=delta_X, V_x=V_x, diff_mat=differential_mat, delta_Y=delta_Y, V_y=V_y)

q = 0
play_interval = 5
Time_series = np.arange(0, T_total, play_interval)
fig = plt.figure(figsize=(15,5))
def animate(i):
    plt.clf()
    plt.subplot(1,2,1)
    plt.contourf(ini_xx, ini_yy, F_qxy_res[0, :, :, i],
                     levels=np.linspace(F_qxy_res[ :, :, :, :].min() - 0.0, F_qxy_res[:,  :, :, :].max(),
                                        80))
    plt.title('t=%s fs; ' % Time_series[i] + 'exciton number: %.2f'%F_qxy_res[ 0, :, :, i].sum())
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.contourf(ini_xx, ini_yy, F_qxy_res[1, :, :, i],
                     levels=np.linspace(F_qxy_res[ :, :, :, :].min() - 0.0, F_qxy_res[:,  :, :, :].max(),
                                        80))

    plt.title('t=%s fs; ' % Time_series[i] + 'exciton number: %.2f'%F_qxy_res[ 1, :, :, i].sum())
    plt.colorbar()


ani = animation.FuncAnimation(fig, animate, np.arange(T_total // play_interval), interval=50)
plt.show()
ani.save('simple_model.gif')

