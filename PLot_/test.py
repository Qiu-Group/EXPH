import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate

f = np.loadtxt("exciton_phonon_mat_cali.dat")
nk = f.shape[0]
# x = f[:,0].reshape((int(np.sqrt(nk)), int(np.sqrt(nk))))
# y = f[:,1].reshape((int(np.sqrt(nk)), int(np.sqrt(nk))))
# z = f[:,3].reshape((int(np.sqrt(nk)), int(np.sqrt(nk))))

# x = f[:,0].reshape((int(np.sqrt(nk)), int(np.sqrt(nk))))
# y = f[:,1].reshape((int(np.sqrt(nk)), int(np.sqrt(nk))))
x = np.linspace(0,1,12)
y = np.linspace(0,1,12)
xx, yy = np.meshgrid(x,y)

z = f[:,3].reshape((int(np.sqrt(nk)), int(np.sqrt(nk))))

f = interpolate.interp2d(xx, yy, z, kind='cubic')

x_new = np.linspace(0,1,144)
y_new = np.linspace(0,1,144)
xx_new, yy_new = np.meshgrid(x_new,y_new)
z_new = f(x_new,y_new)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


surf = ax.plot_surface(xx_new, yy_new, z_new, cmap=cm.cool)
plt.show()