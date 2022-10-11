import numpy as np
from Common.distribution import BE, FD, Dirac_1, Dirac_2
from IO.IO_gkk import read_omega, read_gkk
from IO.IO_acv import read_Acv, read_Acv_exciton_energy
from IO.IO_common import read_bandmap, read_kmap
from Common.kgrid_check import construct_kmap
from ELPH.EL_PH_mat import gqQ
from Common.progress import ProgressBar
from Common.common import move_k_back_to_BZ_1

# calculate the scattering rate
# omega.shape  = (nq, nmode)
# exciton_energy.shape = (nQ, nS)
# acvmat.shape = (nQ,nS,nk,nc,nv,2)
# gkkmat.shape = (nq,nk,ni,nj,nmode)
# gqQ(n_ex_acv_index, m_ex_acv_index, v_ph_gkk, Q_kmap, q_kmap,
#                  acvmat, gkkmat, kmap, kmap_dic, bandmap_occ,muteProgress)


# (1) input
Q_kmap = 6 # this is index of kmap
n_ext_acv_index = 0 # this is index of exciton state in Acv
T = 100 # randomly choosing
degaussian = 0.001

# (2) load and construct map:
acvmat = read_Acv() # load acv matrix
gkkmat =read_gkk() # load gkk matrix
kmap = read_kmap()  # load kmap matrix
bandmap, occ = read_bandmap()  # load band map and number of occupation
kmap_dic = construct_kmap()  # construct kmap dictionary {'k1 k2 k3':[0 0 0 0]}: this is used for mapping final state of scattering
omega_mat = read_omega() # dimension [meV]
exciton_energy = read_Acv_exciton_energy()
h_bar = 6.582119569E-16     # dimension = [eV.s]
# number of point, band and phonon mode
nc = bandmap[:, 0][-1] - occ  # load number of conduction band
nv = occ + 1 - bandmap[:, 0][0]  # load number of valence band
n_phonon = omega_mat.shape[1]
Nqpt = kmap.shape[0] # number of q-points


# (3) calculate scattering rate
# note: kmap.shape(nk, information=(kx,ky,kz,Q, k_acv, q, k_gkk))
progress = ProgressBar(kmap.shape[0], fmt=ProgressBar.FULL) # progress

# loop start with q
collect = []
Gamma_res = 0
factor = 2*np.pi/(h_bar*Nqpt) # dimension = [eV-1.s-1]
for q_kmap in range(kmap.shape[0]):  # q_kmap is the index of kmap from point 0-15 in kmap.dat (e.g)
    progress.current += 1
    progress()
    for m_ext_acv_index_loop in range(exciton_energy.shape[1]): # loop over initial exciton state m
        for v_ph_gkk_index_loop in range(n_phonon): # loop over phonon mode v

            # (1) ex-ph matrix
            # gqQ(n_ex_acv_index, m_ex_acv_index, v_ph_gkk, Q_kmap, q_kmap,
            #                  acvmat, gkkmat, kmap, kmap_dic, bandmap_occ,muteProgress)
            gqQ_sq_temp = np.abs(gqQ(n_ex_acv_index=n_ext_acv_index,
                       m_ex_acv_index=m_ext_acv_index_loop,
                       v_ph_gkk= v_ph_gkk_index_loop,
                       Q_kmap=Q_kmap,
                       q_kmap=q_kmap,
                       acvmat=acvmat,
                       gkkmat=gkkmat,
                       kmap=kmap,
                       kmap_dic=kmap_dic,
                       bandmap_occ= [bandmap,occ],
                       muteProgress=True
                       ))**2 # dimension [eV^2]

            # q_gkk and Q+q_acv index
            # kmapout[x] = [Q, k_acv, q, k_gkk]

            [Q_acv_index, q_gkk_index] = [kmap[Q_kmap, 3], kmap[q_kmap, 5]]
            Q_plus_q_point = move_k_back_to_BZ_1(kmap[Q_kmap, 0:3] + kmap[q_kmap, 0:3])
            Q_plus_q_kmapout = kmap_dic[
                '  %.5f    %.5f    %.5f' % (Q_plus_q_point[0], Q_plus_q_point[1], Q_plus_q_point[2])]
            Qpr_as_Q_plus_q_acv_index = Q_plus_q_kmapout[0]

            # find energy for exciton and phonon (you should use index_acv and index_gkk)
            # omega.shape  = (nq, nmode)
            # exciton_energy.shape = (nQ, nS)
            # OMEGA_xx has included h_bar
            omega_v_q_temp     = omega_mat[int(q_gkk_index),int(v_ph_gkk_index_loop)] * 10 ** (-3) # dimension [eV]
            OMEGA_m_Q_plus_q_temp = exciton_energy[int(Qpr_as_Q_plus_q_acv_index), int(m_ext_acv_index_loop)] # dimension [eV]
            OMEGA_n_Q_temp        = exciton_energy[int(Q_acv_index),               int(m_ext_acv_index_loop)] # dimension [eV]

            # (2) left part
            # print(OMEGA_m_Q_plus_q_temp)
            # todo: check warning: RuntimeWarning: divide by zero encountered
            # print(BE(omega=OMEGA_m_Q_plus_q_temp, T=T))
            # print(OMEGA_m_Q_plus_q_temp)
            distribution_temp = (
                                (BE(omega=omega_v_q_temp, T=T) + 1 + BE(omega=OMEGA_m_Q_plus_q_temp, T=T))
                                 * Dirac_1(OMEGA_n_Q_temp - OMEGA_m_Q_plus_q_temp - omega_v_q_temp, sigma=degaussian)

                                 +

                                (BE(omega=omega_v_q_temp, T=T) - BE(omega=OMEGA_m_Q_plus_q_temp, T=T))
                                * Dirac_1(OMEGA_n_Q_temp - OMEGA_m_Q_plus_q_temp + omega_v_q_temp, sigma=degaussian)

                                 )
            collect.append(factor * gqQ_sq_temp * distribution_temp)

            Gamma_res = Gamma_res + factor * gqQ_sq_temp * distribution_temp

            # (3) right part

collect = np.array(collect)


