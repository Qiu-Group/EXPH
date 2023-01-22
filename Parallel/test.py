import numpy as np
import numpy as np
from Common.progress import ProgressBar
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def Gaussian(x,y,sigma=1,x0=10,y0=10):
    return 1/(2*np.pi*sigma**2) * np.exp(-((x-x0)**2+(y-y0)**2)/(2*sigma**2))


X = 20
Y = 20
nX = 200
nY = 200
T_total = 400
nT = 3200
delta_X = X / nX
delta_Y = Y/ nY
delta_T = T_total / nT

# F_qxy = np.zeros(nX) # .shape = (2,nX,nY)
ini_x = np.arange(0, X, delta_X)
ini_y = np.arange(0, Y, delta_Y)

ini_xx, ini_yy = np.meshgrid(ini_x,ini_y)

F_qxy = Gaussian(ini_xx, ini_yy) *0.5 # Initial Ccondition
F_qxy_res = np.zeros((nX,nY,nT)) # .shape = (2,nX,nY,nT)
V_x = 0.
V_y = 0.1

# (1) Differential Matrix: (Euler Forward)
# C = V_x*delta_T/delta_X
# differential_mat = (-1*np.eye(nX) + np.eye(nX,k=-1)) * C

# (2) Lax-Wendroff
C_x = V_x*delta_T/delta_X
a1 = -1*C_x*(1-C_x)/2
a_neg1 = C_x*(1+C_x)/2
a0 = -C_x**2

differential_mat_x = np.eye(nX,k=-1) * a_neg1  + np.eye(nX) *a0 + np.eye(nX, k=1) * a1
differential_mat_x[-1,0] =  a1
differential_mat_x[0,-1] = a_neg1

C_y = V_y*delta_T/delta_Y
a1 = -1*C_y*(1-C_y)/2
a_neg1 = C_y*(1+C_y)/2
a0 = -C_y**2

differential_mat_y = np.eye(nY,k=-1) * a_neg1  + np.eye(nY) *a0 + np.eye(nY, k=1) * a1
differential_mat_y[-1,0] =  a1
differential_mat_y[0,-1] = a_neg1


progress = ProgressBar(nT, fmt=ProgressBar.FULL)
for i in range(nT):
    progress.current += 1
    progress()
    f = np.matmul(F_qxy, differential_mat_x) + np.matmul(differential_mat_y, F_qxy)
    F_qxy = F_qxy + f
    F_qxy_res[:,:, i] = F_qxy




play_interval = 2
# Time_series = np.arange(0, T_total, play_interval)
fig = plt.figure(figsize=(8,5))
def animate(i):
    plt.clf()

    # # plt.plot(ini_x,F_qxy_res[:,i])
    # # plt.ylim(-0.02,0.1)
    # # plt.axhline(y=F_qxy_res[:,0].max(),color='r')
    #
    # plt.title('t=%s fs; ' % int(i * delta_T) + 'exciton number: %.2f'%F_qxy_res[ :, i].sum() + ' i: %s'%i)
    # # plt.colorbar()
    plt.contourf(ini_xx, ini_yy, F_qxy_res[:, :, i], levels=np.linspace( -0.02, 0.1,80))

    plt.title('t=%s fs; ' % int(i * delta_T) + 'exciton number: %.2f'%F_qxy_res[  :, :, i].sum())
    plt.colorbar()


ani = animation.FuncAnimation(fig, animate, np.arange(0, nT, int(play_interval/delta_T)), interval=50)
plt.show()