import numpy as np
from Common.progress import ProgressBar
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class LinearAdvection1D:
    # Matrix for LA1D
    A = 0

    # Initialization of constants
    def __init__(self, c, x0, xN, N, deltaT, T):
        self.c = c
        self.x0 = x0
        self.xN = xN
        self.N = N
        self.deltaT = deltaT
        self.T = T
        # CFL number funct.

    def CFL(self):
        deltaX = (self.xN - self.x0) / self.N
        return self.c * self.deltaT / deltaX

    # check CFL number <=1 or not.
    def checkCFL(self):
        if (np.abs(self.CFL()) <= 1):
            flag = True
        else:
            flag = False
        return flag

    # Matrix assembly of LA1D
    def upwindMatrixAssembly(self):
        alpha_min = min(self.CFL(), 0)
        alpha_max = max(self.CFL(), 0)
        a1 = [alpha_max] * (self.N - 1)
        a2 = [1 + alpha_min - alpha_max] * (self.N)
        a3 = [-alpha_min] * (self.N - 1)
        self.A = np.diag(a1, -1) + np.diag(a2, 0) + np.diag(a3, 1)
        self.A[0, -1] = alpha_max
        self.A[N - 1, 0] = -alpha_min

    def LaxWendroffMatrixAssembly(self):
        C = self.CFL()
        aleft = C * (1 + C) / 2.
        amid = 1. - (C * C)
        aright = C * (C - 1) / 2.
        a1 = [aleft] * (self.N - 1)
        a2 = [amid] * (self.N)
        a3 = [aright] * (self.N - 1)
        self.A = np.diag(a1, -1) + np.diag(a2, 0) + np.diag(a3, 1)
        self.A[0, -1] = aleft
        self.A[N - 1, 0] = aright

        # Solve u=Au0
    def Solve(self, u0):
        return np.matmul(self.A, u0)

    #############


# Start of the code
###################

# constants
N, x0, xN, deltaT, c, T = 100, 0., 10., 0.02, 0.3, 200
# initialization of constants
LA1D = LinearAdvection1D(c, x0, xN, N, deltaT, T)
u_res_lax = np.zeros((N,int(LA1D.T / LA1D.deltaT)))
u_res_euler = np.zeros((N,int(LA1D.T / LA1D.deltaT)))

# initial value
x = np.linspace(LA1D.x0, LA1D.xN, LA1D.N)
u0 = np.exp(-(x - 2) * (x - 2))

# calculating solution if CFL<=1
if (LA1D.checkCFL() is True):
    print("CFL number is: ", LA1D.CFL())
    # LA1D.upwindMatrixAssembly()
    LA1D.LaxWendroffMatrixAssembly()
    for t in range(0, int(LA1D.T / LA1D.deltaT)):
        u_res_lax[:,t] = u0
        u = LA1D.Solve(u0)
        u0 = u

else:
    print("CFL number is greater than 1. CFL: ", LA1D.CFL())

u0 = np.exp(-(x - 2) * (x - 2))
if (LA1D.checkCFL() is True):
    print("CFL number is: ", LA1D.CFL())
    # LA1D.upwindMatrixAssembly()
    LA1D.upwindMatrixAssembly()
    for t in range(0, int(LA1D.T / LA1D.deltaT)):
        u_res_euler [:,t] = u0
        u = LA1D.Solve(u0)
        u0 = u

else:
    print("CFL number is greater than 1. CFL: ", LA1D.CFL())


play_interval = 1
# Time_series = np.arange(0, T, play_interval)
fig = plt.figure(figsize=(10,5))

def animate(i):
    plt.clf()
    plt.axhline(y=u_res_lax[:,0].max(),color='r')
    plt.plot(x,u_res_lax[:,i])
    plt.plot(x, u_res_euler[:, i])
    plt.ylim(-0.2,2)

    plt.title('t=%s fs; ' % int(i * deltaT))
    # plt.colorbar()


ani = animation.FuncAnimation(fig, animate, np.arange(0,int(LA1D.T / LA1D.deltaT), int(play_interval/LA1D.deltaT)), interval=200)
plt.show()

