import numpy as np
import warnings
# todo: check if nmode should be 1 for BE and FD

# def BE(omega, T, mode_num=1):
#     # tododone: check this!!! u should be 0, but some times, it goes diverence when omega = 0
#     u = -0.1
#     K_B = 8.617343e-05  ### Dimension: eV/K
#     with np.errstate(divide='ignore'):
#         return (mode_num / (np.exp((omega  - u) / (K_B * T)) - 1))

def BE(omega, T, mode_num=1):
    u = 0.0
    K_B = 8.617343e-05  ### Dimension: eV/K
    # TODO: Check this!!!
    warnings.filterwarnings('ignore')
    return (mode_num / (np.exp((omega - u) / (K_B * T)) - 1))


def FD(omega, u_f, T, mode_num=1):
    K_B = 8.617343e-05  ### Dimension: eV/K
    return (mode_num / (np.exp((omega  - u_f) / (K_B * T)) + 1))

def Dirac_1(x,sigma):
    """
    Gaussian method
    :param x: input
    :param sigma: the variance of Gaussian function
    :return:
    """
    return np.exp( (-1/2) * (x/sigma)**2 ) * (1/sigma)

def Dirac_2(x, tolerance):
    """
    :param x: input
    :param tolerance: within the tolerance, we think the energy is actually the same
    :return:
    """
    if abs(x) < tolerance:
        return 1
    else:
        return 0
