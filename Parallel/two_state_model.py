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

def solve_it(F_qxy_res,F_qxy, dfdt_res,diff_mat, delta_T, diff_mat_y):
    progress = ProgressBar(nT, fmt=ProgressBar.FULL)
    for i in range(nT):
        progress.current += 1
        progress()

        dfdt = update_F_nQq(F_qxy_last=F_qxy, g_qp=g_qp)
        dfdt_res[:,:,:,i] = dfdt[:,:,:]
        F_qxy_res[:, :, :, i] = F_qxy[:, :, :]

        F_qxy = F_qxy \
                + np.matmul(F_qxy, diff_mat)  \
                + np.matmul(diff_mat_y,F_qxy )  \
                + delta_T * dfdt * delta_T  \


X, Y = 20,20
nX, nY = 80, 80
T_total = 40
delta_T = 0.02
nT = int(T_total/delta_T)

delta_X, delta_Y = X / nX , Y / nY
delta_T = T_total / nT

ini_x = np.arange(0, X, delta_X)
ini_y = np.arange(0, Y, delta_Y)
ini_xx, ini_yy = np.meshgrid(ini_x, ini_y)

g_qp = np.array([[0,10,10,30,30],[10,0,0,10,10],[10,0,0,10,10],[30,10,10,0,0],[30,10,10,0,0]])*0.5 # .shape = (2,2)
F_qxy = np.zeros((5,nX,nY)) # .shape = (2,nX,nY)
F_qxy[0] = Gaussian(ini_xx,ini_yy)  # Initial Ccondition
F_qxy_res = np.zeros((5,nX,nY,nT)) # .shape = (2,nX,nY,nT)
dfdt_res = np.zeros((5,nX,nY,nT))
V_x = np.array([0., 0.1, -0.1, 0.2, -0.2])[:,np.newaxis,np.newaxis] # .shape(2,1,1)
V_y = np.array([0., 0.2, -0.2, 0. , 0.])[:,np.newaxis,np.newaxis] # .shape(2,1,1)

# Euler Forward
# differential_mat = -1*np.eye(nX) + np.eye(nX,k=-1)

# Lax
C = V_x*delta_T/delta_X
a1 = -1*C*(1-C)/2
a_neg1 = C*(1+C)/2
a0 = -C**2

differential_mat = np.eye(nX,k=-1) * a_neg1  + np.eye(nX) *a0 + np.eye(nX, k=1) * a1
differential_mat[:,-1,0] = differential_mat[:,-1,0] +  a1[:,0,0]
differential_mat[:,0,-1] = differential_mat[:,-1,0] + a_neg1[:,0,0]


C_y = V_y*delta_T/delta_Y
a1 = -1*C_y*(1-C_y)/2
a_neg1 = C_y*(1+C_y)/2
a0 = -C_y**2

differential_mat_y = np.eye(nY,k=-1) * a_neg1  + np.eye(nY) *a0 + np.eye(nY, k=1) * a1
differential_mat_y[:,-1,0] = differential_mat[:,-1,0] +  a1[:,0,0]
differential_mat_y[:,0,-1] = differential_mat[:,-1,0] + a_neg1[:,0,0]


# Boundary Condition:
# differential_mat[:, 0] = differential_mat[:, -1] = 0.1
# differential_mat[0, :] = differential_mat[-1, :] = 0.1


# solve and plot
solve_it(F_qxy_res=F_qxy_res, F_qxy=F_qxy, dfdt_res=dfdt_res, delta_T=delta_T,  diff_mat=differential_mat, diff_mat_y = differential_mat_y)

q = 0
play_interval = 1
fig = plt.figure(figsize=(15,8))
def animate(i):
    plt.clf()
    plt.subplot(2,3,1)
    # plt.contourf(ini_xx, ini_yy, F_qxy_res[0, :, :, i],
    #                  levels=np.linspace(F_qxy_res[ :, :, :, :].min() - 0.0, F_qxy_res[:,  :, :, :].max(),
    #                                     80))
    plt.contourf(ini_xx, ini_yy, F_qxy_res[0, :, :, i], levels=np.linspace( -0.01, 0.16,80))

    plt.title('t=%s fs; ' % int(i * delta_T) + 'exciton number: %.2f'%F_qxy_res[ 0, :, :, i].sum())
    plt.colorbar()

    plt.subplot(2,3,2)
    plt.contourf(ini_xx, ini_yy, F_qxy_res[1, :, :, i], levels=np.linspace( -0.01, 0.06,80))

    plt.title('t=%s fs; ' % int(i * delta_T) + 'exciton number: %.2f'%F_qxy_res[ 1, :, :, i].sum())
    plt.colorbar()

    plt.subplot(2,3,3)
    plt.contourf(ini_xx, ini_yy, F_qxy_res[2, :, :, i], levels=np.linspace( -0.01, 0.06,80))

    plt.title('t=%s fs; ' % int(i * delta_T) + 'exciton number: %.2f'%F_qxy_res[ 2, :, :, i].sum())
    plt.colorbar()

    plt.subplot(2,3,5)
    plt.contourf(ini_xx, ini_yy, F_qxy_res[3, :, :, i], levels=np.linspace( -0.01, 0.06,80))

    plt.title('t=%s fs; ' % int(i * delta_T) + 'exciton number: %.2f'%F_qxy_res[ 2, :, :, i].sum())
    plt.colorbar()

    plt.subplot(2,3,6)
    plt.contourf(ini_xx, ini_yy, F_qxy_res[4, :, :, i], levels=np.linspace( -0.01, 0.06,80))

    plt.title('t=%s fs; ' % int(i * delta_T) + 'exciton number: %.2f'%F_qxy_res[ 2, :, :, i].sum())
    plt.colorbar()

ani = animation.FuncAnimation(fig, animate, np.arange(0, nT, int(play_interval/delta_T)), interval=100)
plt.show()
ani.save('simple_model_200.htm')

