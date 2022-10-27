import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate

def interqp_2D(mats, interpo_size, kind='linear'):
    """
    :param matx: the matrix to be interpolated
    :param interpo_size: final size after interpolated: interpo_size * interpo_size
    :return: new mats
    """
    try:
        if mats.shape[0] != mats.shape[1]:
            raise Exception("mats.shape[0] != mats.shape[1]")
    except:
        raise Exception("interpolation: mats size issue")
    nk = mats.shape[0]
    X_org = np.linspace(0, 1, nk)
    Y_org = np.linspace(0, 1, nk)
    # XX, YY = np.meshgrid(X_org, Y_org)

    X_new = np.linspace(0,1, interpo_size)
    Y_new = np.linspace(0,1, interpo_size)

    f_z = interpolate.interp2d(X_org, Y_org, mats, kind=kind)
    z_new = f_z(X_new, Y_new)
    return z_new