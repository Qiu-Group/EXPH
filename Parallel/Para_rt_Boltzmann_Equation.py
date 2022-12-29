from ELPH.EX_PH_mat import gqQ
from Parallel.Para_common import plan_maker, before_parallel_job, after_parallel_sum_job
from mpi4py import MPI
from IO.IO_common import construct_kmap, read_kmap, read_bandmap, readkkqQ
from IO.IO_gkk import read_gkk
from IO.IO_acv import read_Acv
from time import  process_time
import time
from IO.IO_common import read_kmap, read_lattice
from Common.common import frac2carte
# from Common.progress import ProgressBar
import numpy as np
import h5py as h5
from IO.IO_gkk import read_omega, read_gkk
from IO.IO_acv import read_Acv, read_Acv_exciton_energy
from Common.common import move_k_back_to_BZ_1
from Common.distribution import Dirac_1, BE
from Common.progress import ProgressBar
import matplotlib.pyplot as plt
import matplotlib.animation as animation

deguassian = 0.03
path = '../'
gqQ_path = path + 'gqQ.h5'
T = 500

bvec = read_lattice('b', path)
acvmat = read_Acv(path=path)
gkkmat = read_gkk(path=path)
kmap = read_kmap(path=path)
kmap_dic = construct_kmap(path=path)
bandmap_occ = read_bandmap(path=path)
exciton_energy = read_Acv_exciton_energy(path=path)
omega_mat = read_omega(path=path)  # dimension [meV]
h_bar = 6.582119569E-16  # dimension = [eV.s]
Constant = -1*(2*np.pi/h_bar) * (1/int(kmap.shape[0]))


def update_F_nQq(F_nQ):
    """
    :param F_nQ:
    :return: F_nQq.shape = (n,Q,q)
    """
    time1 = time.time()
    Q_acv_back2_kmap = list(map(int,list(kmap[:,3])))
    F_nQq = np.zeros((F_nQ.shape[0],F_nQ.shape[1],F_nQ.shape[1]))
    for i_Q_kmap in range(kmap.shape[0]):
        for i_q_kmap in range(kmap.shape[0]):
            F_nQq[:,i_Q_kmap,i_q_kmap] = F_nQ[:,Q_acv_back2_kmap.index(Qplusq_2_QPrimeIndexAcv(i_Q_kmap,i_q_kmap))]
    time2 = time.time()
    # print('t: ',time2-time1)
    return F_nQq


def Qplusq_2_QPrimeIndexAcv(Q_kmap, q_kmap):
    """
    :param Q_kmap: index of Q from 0-nQ in kmap
    :param q_kmap: index of q from 0-nq in kmap
    :return: Qpr_as_Q_plus_q_acv_index
    """
    Q_plus_q_point = move_k_back_to_BZ_1(kmap[Q_kmap, 0:3] + kmap[q_kmap, 0:3])
    key_temp = '  %.5f    %.5f    %.5f' % (Q_plus_q_point[0], Q_plus_q_point[1], Q_plus_q_point[2])
    Q_plus_q_kmapout = kmap_dic[key_temp.replace('-', '')]
    Qpr_as_Q_plus_q_acv_index = Q_plus_q_kmapout[0]
    return int(Qpr_as_Q_plus_q_acv_index)
    pass

def Construct_Delta(deguassian):
    # E.shape = (n,m,v,Q,q)
    E_nQ = get_E_nQ()[:,np.newaxis,np.newaxis,:,np.newaxis]
    E_mQq = get_E_mQq()[np.newaxis,:,np.newaxis,:,:]
    E_vq = get_E_vq()[np.newaxis,np.newaxis,:,np.newaxis,:]

    Delta_pos_nmvQq = Dirac_1(x=E_nQ-E_mQq+E_vq, sigma=deguassian)
    Delta_neg_nmvQq = Dirac_1(x=E_nQ-E_mQq-E_vq, sigma=deguassian)
    return Delta_pos_nmvQq, Delta_neg_nmvQq

def get_E_nQ():
    """
    Q = Q_kmap; n = acv_map
    :return: E_nQ.shape = (n,Q)
    """
    Omega_nQ = exciton_energy.T
    Q_acv_index_list = kmap[:,3]
    E_nQ = Omega_nQ[:n,list(map(int,list(Q_acv_index_list)))]
    return E_nQ

def get_E_vq():
    """
    q = q_kmap; v =gkk_map
    :return: E_vq.shape = (v,q)
    """
    omega_vq = omega_mat.T
    q_gkk_index_list = kmap[:,5]
    E_vq = omega_vq[:v,list(map(int,list(q_gkk_index_list)))]
    return E_vq * 10 ** (-3)

def get_E_mQq():
    """
    :return:E_mQq.shape = (m,Q,q)
    """
    Omega_nQ = exciton_energy.T
    E_mQq = np.zeros((1,m,1,kmap.shape[0],kmap.shape[0]))
    for i_Q_kmap in range(kmap.shape[0]):
        for i_q_kmap in range(kmap.shape[0]):
            E_mQq[0,:,0,i_Q_kmap,i_q_kmap] = Omega_nQ[:m,Qplusq_2_QPrimeIndexAcv(i_Q_kmap,i_q_kmap)]
    return E_mQq[0,:,0,:,:]

def rhs_Fermi_Goldenrule(F_nQ, N_vq, gqQ_mat, Delta_positive, Delta_negative):
    F_nQq = update_F_nQq(F_nQ)
    F_abs = np.einsum('np,vq,npq->npqv', F_nQ, N_vq, 1 + F_nQq) - np.einsum('np,vq,npq->npqv', 1 + F_nQ, 1 + N_vq, F_nQq)
    F_em = np.einsum('np,vq,npq->npqv', F_nQ, 1 + N_vq, 1 + F_nQq) - np.einsum('np,vq,npq->npqv', 1 + F_nQ, N_vq, F_nQq)
    dFdt =  -1*(np.einsum('pqnmv,nmvpq,npqv->np', gqQ_mat, Delta_positive, F_abs) + np.einsum('pqnmv,nmvpq,npqv->np', gqQ_mat, Delta_negative,F_em))
    return  dFdt


nT = 30000
T_total = 30000 #fs
delta_T = T_total/nT

# G(Q_kmap, q_kmap, n, m, v)
# f = h5.File('gqQ.h5','r')
# gqQ_mat = f['data'][()]
f = h5.File('gqQ.h5','r')
gqQ_mat = f['data'][()]
f.close()

# v = omega_mat.shape[1]
# n = exciton_energy.shape[1]
# m = exciton_energy.shape[1]
# Q = int(kmap.shape[0])
# q = int(kmap.shape[0])

v = gqQ_mat.shape[4]
m = gqQ_mat.shape[3]
n = gqQ_mat.shape[2]
q = gqQ_mat.shape[1]
Q = gqQ_mat.shape[0]

# Note!!!: All Q and q here are Q_kmap and q_kmap.
# 1. Initialize
# (1) This part doesn't need to be updated
# Dealta_positve = np.ones((n, m, v, Q, q))
# Dealta_negative = np.ones((n, m, v, Q, q))
# G(Q_kmap, q_kmap, n, m, v)
# f = h5.File('gqQ.h5','r')
# gqQ_mat = f['data'][()]
# f.close()
Delta_positive, Delta_negative = Construct_Delta(deguassian)

# (2) This part needs to be updated for each time step.
# F_nQ = np.ones((n,Q))
# N_vq = np.ones((v,q))
# F_nQq = np.ones((n,Q,q))

E_nQ = get_E_nQ()
F_nQ = BE(omega=E_nQ,T=T)
F_nQ[2,0] = 0.5 # initialize

F_nQq = update_F_nQq(F_nQ)

E_vq = get_E_vq()
N_vq = BE(omega=E_vq,T=T)
N_vq[0:3,0] = np.array([0,0,0])
# 2. Iterate:

# (3) F_abs/em.shape = (n,Q,q,v)
# F_abs = np.einsum('np,vq,npq->npqv',F_nQ,N_vq,1+F_nQq) - np.einsum('np,vq,npq->npqv',1+F_nQ,1+N_vq,F_nQq)
# F_em = np.einsum('np,vq,npq->npqv',F_nQ,1+N_vq,1+F_nQq) - np.einsum('np,vq,npq->npqv',1+F_nQ,N_vq,F_nQq)
# # (4) dF/dt
# dFdt = Constant * (np.einsum('pqnmv,nmvpq,npqv->np',gqQ_mat,Dealta_positve,F_abs) + np.einsum('pqnmv,nmvpq,npqv->np',gqQ_mat,Dealta_negative,F_em))

progress = ProgressBar(nT, fmt=ProgressBar.FULL)
F_nQ_res = np.zeros((n,Q,nT))
exciton_number  = np.zeros(nT)
dfdt_sum_res = np.zeros(nT)
damping_term = np.zeros((n,Q))
for it in range(nT):
    damping_term[:,0] = F_nQ[:,0]
    progress.current += 1
    progress()
    F_nQ_res[:,:,it] = F_nQ
    dfdt = rhs_Fermi_Goldenrule(F_nQ, N_vq, gqQ_mat, Delta_positive, Delta_negative)
    error_from_nosymm = dfdt.sum()/(dfdt.shape[0]*dfdt.shape[1])

    F_nQ = F_nQ +  (dfdt - error_from_nosymm) * delta_T - damping_term * delta_T * 0.1

    # some debugging
    exciton_number[it] = F_nQ.sum()
    dfdt_sum_res[it] = dfdt.sum()
    # print(dfdt[2,0])


# plt.plot(np.linspace(1,T_total,nT),F_nQ_res[1,0,:])
# plt.plot(np.linspace(1,T_total,nT),exciton_number)


fig = plt.figure()
Q_exciton = np.arange(Q) * np.ones((n,Q))
enegy = E_nQ
plt.scatter(Q_exciton,enegy)

# Time_seriers = np.linspace(1,T_total,nT)
T_interval = 200
Time_series = np.arange(0,T_total,T_interval)
def animate(i):
    plt.clf()
    plt.scatter(Q_exciton,enegy,s=F_nQ_res[:,:,i]*50000)
    plt.title(label='t=%s fs'%Time_series[i])
    # plt.bar(box, ParticleNumerInBox[:, i], width=width,label="Diffusion: t=%s s" % T[i])
    # plt.bar(box+width, ParticleNumerInBox_MC[:,i],width=width,label="Random Walk: t=%s s"%T[i])
    # plt.ylim(0,100)
    plt.xlabel("Q")
    plt.ylabel("Energy")
    plt.ylim([2,3])
    # plt.legend()

ani = animation.FuncAnimation(fig,animate,np.arange(T_total//T_interval),interval=10)
plt.show()
ani.save('test.gif')