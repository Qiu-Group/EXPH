import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from Common.inteqp import interqp_2D

interpo_size =  64
#
f = np.loadtxt("exciton_phonon_mat.dat")
nk = f.shape[0]
#
# X = np.linspace(0,1,int(np.sqrt(nk)))
# Y = np.linspace(0,1,int(np.sqrt(nk)))
# XX, YY = np.meshgrid(X,Y)
#
xx = f[:,0].reshape((int(np.sqrt(nk)), int(np.sqrt(nk))))
yy = f[:,1].reshape((int(np.sqrt(nk)), int(np.sqrt(nk))))
z = f[:,3].reshape((int(np.sqrt(nk)), int(np.sqrt(nk))))
#
# # (1) interpolate x and y by X and Y
# f_xx = interpolate.interp2d(XX,YY,xx, kind='linear')
# f_yy = interpolate.interp2d(XX,YY,yy, kind='linear')
# X_new = np.linspace(0,1, interpo_size)
# Y_new = np.linspace(0,1, interpo_size)
# xx_new = f_xx(X_new,Y_new)
# yy_new = f_yy(X_new,Y_new)
#
# # (2) interpolate z by X and Y
# f_z = interpolate.interp2d(XX,YY,z, kind='linear')
# z_new = f_z(X_new,Y_new)

xx_new = interqp_2D(xx,interpo_size=interpo_size)
yy_new = interqp_2D(yy,interpo_size=interpo_size)
z_new = interqp_2D(z,interpo_size=interpo_size)


# x = f[:,0].reshape((int(np.sqrt(nk)), int(np.sqrt(nk))))
# y = f[:,1].reshape((int(np.sqrt(nk)), int(np.sqrt(nk))))
# x = np.linspace(0,1,12)
# y = np.linspace(0,1,12)
# xx, yy = np.meshgrid(x,y)
#
# z = f[:,3].reshape((int(np.sqrt(nk)), int(np.sqrt(nk))))
#
#
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(xx, yy, z, cmap=cm.cool)
# plt.show()
#
#
#
# f = interpolate.interp2d(xx, yy, z, kind='cubic')
# x_new = np.linspace(0,1,24)
# y_new = np.linspace(0,1,24)
# xx_new, yy_new = np.meshgrid(x_new,y_new)
# z_new = f(x_new,y_new)
#
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#
#
surf = ax.plot_surface(xx_new, yy_new, z_new, cmap=cm.cool)
plt.show()