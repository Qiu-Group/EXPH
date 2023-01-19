import numpy as np
import numpy as np
from Common.progress import ProgressBar
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def Gaussian(x,sigma=1,x0=10):
    return 1/(2*np.pi*sigma**2) * np.exp(-((x-x0)**2)/(2*sigma**2))


X = 20
nX = 200
T_total = 400
nT = 12800
delta_X = X / nX
delta_T = T_total / nT

# F_qxy = np.zeros(nX) # .shape = (2,nX,nY)
ini_x = np.arange(0, X, delta_X)
F_qxy = Gaussian(ini_x) *0.5 # Initial Ccondition
F_qxy_res = np.zeros((nX,nT)) # .shape = (2,nX,nY,nT)
V_x = 0.1

# (1) Differential Matrix: (Euler Forward)
# C = V_x*delta_T/delta_X
# differential_mat = (-1*np.eye(nX) + np.eye(nX,k=-1)) * C

# (2) Lax-Wendroff
C = V_x*delta_T/delta_X
a1 = -1*C*(1-C)/2
a_neg1 = C*(1+C)/2
a0 = -C**2

differential_mat = np.eye(nX,k=-1) * a_neg1  + np.eye(nX) *a0 + np.eye(nX, k=1) * a1
differential_mat[-1,0] =  a1
differential_mat[0,-1] = a_neg1


progress = ProgressBar(nT, fmt=ProgressBar.FULL)
for i in range(nT):
    progress.current += 1
    progress()
    f = np.matmul(F_qxy, differential_mat)
    F_qxy = F_qxy + f
    F_qxy_res[:, i] = F_qxy




play_interval = 1
# Time_series = np.arange(0, T_total, play_interval)
fig = plt.figure(figsize=(15,5))
def animate(i):
    plt.clf()

    plt.plot(ini_x,F_qxy_res[:,i])
    plt.ylim(-0.02,0.1)
    plt.axhline(y=F_qxy_res[:,0].max(),color='r')

    plt.title('t=%s fs; ' % int(i * delta_T) + 'exciton number: %.2f'%F_qxy_res[ :, i].sum() + ' i: %s'%i)
    # plt.colorbar()


ani = animation.FuncAnimation(fig, animate, np.arange(0, nT, int(play_interval/delta_T)), interval=50)
plt.show()