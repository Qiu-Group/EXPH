import numpy as np
from IO.IO_common import read_bandmap
from Common.kgrid_check import construct_kmap
from IO.IO_acv import read_Acv
from IO.IO_gkk import read_gkk
from IO.IO_common import read_bandmap, read_kmap
from Common.common import move_k_back_to_BZ_1
from Common.h5_status import check_h5_tree
import h5py as h5
from Common.progress import ProgressBar



# def gqQ(n_ex_acv, m_ex_acv, v_ph_gkk, Q_kmap, q_kmap, Acv=, gkk):
"""
This function construct gnmv(Q,q)
:param n_ex_acv: index of initial exciton state
:param m_ex_acv: index of final exciton state
:param v_ph_gkk: index of phonon mode
:param Q_kmap: exciton momentum
:param q_kmap: phonon momentumB
:param Acv: acv matrix (do not read every time)
:return:
"""

# input (as variable)
# !!! Just two example, you need to intensively test on this
Q_kmap = 6 # this is index of kmap
q_kmap = 12 # this is index of kmap
n_ex_acv_index = 0 # this is index of exciton state in Acv
m_ex_acv_index = 0 # this is index of exciton state in Acv
v_ph_gkk = 3 # this is index of phonon mode

# loading acv, gkk, kmap and bandmap
acvmat = read_Acv() # load acv matrix
gkkmat =read_gkk() # load gkk matrix
kmap = read_kmap() # load kmap matrix
bandmap, occ = read_bandmap() # load band map and number of occupation
nc = bandmap[:,0][-1] - occ # load number of conduction band
nv = occ + 1 - bandmap[:,0][0] # load number of valence band
kmap_dic = construct_kmap() # construct kmap dictionary {'k1 k2 k3':[0 0 0 0]}: this is used for mapping final state of scattering

# acv = h5.File("Acv.h5",'r')
# acv["mf_header/kpoints"]

progress = ProgressBar(kmap.shape[0], fmt=ProgressBar.FULL)
# note: kmap.shape(nk, information=(kx,ky,kz,Q, k_acv, q, k_gkk))
for k_kmap in range(kmap.shape[0]):  # k_kmap is the index of kmap from 0-15 (e.g)
    # get the right index in acv and gkk
    # print("k_acv_index: %s, k_acv: " %k_acv_index, acv["exciton_header/kpoints/kpt_for_each_Q"][k_acv_index])
    # print("k_acv_index: %s "%i_kmap)
    progress.current += 1
    progress()
    res = np.complex(0,0)

    first_res = 0
    second_res = 0

    # I. first part of equation (5) in Bernardi's paper
    # first_res
    for v_bandmap in range(int(nv)): # start with the lowest valence band
        # print("v_bandmap:",v_bandmap)
        for c_bandmap in range(int(nv),int(nv +nc)): # start with the lowest conduction band
            # print("c_bandmap:",c_bandmap)
            for cpr_bandmap in range(int(nv),int(nv +nc)): # start with the lowest conduction band

                # 1.0 get the right index for band and k-points in acv and gkk MATRIX
                # kmap.shape(nk, information=(kx, ky, kz, Q, k_acv, q, k_gkk))
                # todo: move Q and q out of loop
                [Q_acv_index, q_gkk_index] = [ kmap[Q_kmap, 3], kmap[q_kmap,5]]
                [k_acv_index, k_gkk_index] = [ kmap[k_kmap, 4], kmap[k_kmap, 6]]
                [v_acv_index, c_acv_index, cpr_acv_index] = [bandmap[v_bandmap,1], bandmap[c_bandmap,1], bandmap[cpr_bandmap,1]]
                [v_gkk_index, c_gkk_index, cpr_gkk_index] = [bandmap[v_bandmap,2], bandmap[c_bandmap,2], bandmap[cpr_bandmap,2]]
                # print(v_acv,c_acv,cpr_acv)
                # print("cpr_bandmap:",cpr_bandmap)

                # 2.0 we need to find these new k/q/Q in the 1s BZ:
                # res1 <- [Q+q, k+Q+q, k+Q]; res2 <- [Q+q, k-q, k+Q]
                # todo: actually, we can move Q+q out of this loop, since this is actually a constan for the given Q and q
                # kmap_dic = {'  %.6f    %.6f    %.6f' : [Q, k_acv, q, k_gkk], ...}
                # kmapout[x] = [Q, k_acv, q, k_gkk]
                Q_plus_q_point =  move_k_back_to_BZ_1(kmap[Q_kmap,0:3] + kmap[q_kmap,0:3])
                Q_plus_q_kmapout = kmap_dic['  %.6f    %.6f    %.6f'%(Q_plus_q_point[0], Q_plus_q_point[1], Q_plus_q_point[2])]
                Q_plus_q_plus_k_point = move_k_back_to_BZ_1(Q_plus_q_point + kmap[k_kmap,0:3])
                Q_plus_q_plus_k_kmapout = kmap_dic['  %.6f    %.6f    %.6f'%(Q_plus_q_plus_k_point[0], Q_plus_q_plus_k_point[1], Q_plus_q_plus_k_point[2])]
                k_plus_Q_point = move_k_back_to_BZ_1(kmap[k_kmap,0:3] + kmap[Q_kmap,0:3])
                k_plus_Q_kmapout = kmap_dic['  %.6f    %.6f    %.6f'%(k_plus_Q_point[0], k_plus_Q_point[1], k_plus_Q_point[2])]

                # 3.0 Calculation!
                # acvmat.shape = (nQ,nS,nk,nc,nv,2)
                # gkkmat.shape = (nq,nk,ni,nj,nmode) # ni is initial, ni is final

                # (i) Acv_temp1
                # this is k'=k+Q+q and Q'=Q+q for Acv_temp1:
                # it seems there is no need to find k_pr (think more...)
                # kpr_as_Q_plus_q_plus_k_acv = Q_plus_q_plus_k_kmapout[1] # 1 means the fifth column of kmap, which is k_acv
                # kmapout[x] = [Q, k_acv, q, k_gkk]
                Qpr_as_Q_plus_q_acv_index = Q_plus_q_kmapout[0] # 0 means the fourth column of kmap, which is Q
                Acv_temp1 = np.complex(
                    acvmat[int(Qpr_as_Q_plus_q_acv_index), int(m_ex_acv_index), int(k_acv_index), int(c_acv_index), int(v_acv_index)][0],
                    acvmat[int(Qpr_as_Q_plus_q_acv_index), int(m_ex_acv_index), int(k_acv_index), int(c_acv_index), int(v_acv_index)][1]
                )
                Acv_temp1_conjugated = np.conjugate(Acv_temp1)

                # (ii) Acv_temp2
                # this is Q for Acv_temp2 (Q_acv_index has been found in previous step), so we can start to do Acv_temp2
                Acv_temp2 = np.complex(
                    acvmat[int(Q_acv_index), int(n_ex_acv_index), int(k_acv_index), int(cpr_acv_index), int(v_acv_index)][0],
                    acvmat[int(Q_acv_index), int(n_ex_acv_index), int(k_acv_index), int(cpr_acv_index), int(v_acv_index)][1]
                )
                # (iii) gkk_temp
                # this is kpr = k + Q
                # kmapout[x] = [Q, k_acv, q, k_gkk]
                kpr_as_k_plus_Q_gkk_index = k_plus_Q_kmapout[3] # 0 means the seventh column of kmap, which is k_gkk
                gkk_temp = gkkmat[int(q_gkk_index), int(kpr_as_k_plus_Q_gkk_index),int(c_gkk_index),int(cpr_gkk_index),int(v_ph_gkk)] # initial is c ->i; final is c' -> j

                # (iv) res_temp
                first_res = first_res + Acv_temp1_conjugated * Acv_temp2 * gkk_temp


    # II. second part of equation (5) in Bernardi's paper
    # second_res
    for v_bandmap in range(int(nv)):
        # print("v_bandmap:",v_bandmap)
        for c_bandmap in range(int(nv),int(nv +nc)):
            # print("c_bandmap:",c_bandmap)
            for vpr_bandmap in range(int(nv)):

                # 1.0 get the right index for band and k-points in acv and gkk MATRIX
                # kmap.shape(nk, information=(kx, ky, kz, Q, k_acv, q, k_gkk))
                # todo: move Q and q out of loop
                [Q_acv_index, q_gkk_index] = [kmap[Q_kmap, 3], kmap[q_kmap, 5]]
                [k_acv_index, k_gkk_index] = [kmap[k_kmap, 4], kmap[k_kmap, 6]]
                [v_acv_index, c_acv_index, vpr_acv_index] = [bandmap[v_bandmap,1], bandmap[c_bandmap,1], bandmap[vpr_bandmap,1]]
                [v_gkk_index, c_gkk_index, vpr_gkk_index] = [bandmap[v_bandmap,2], bandmap[c_bandmap,2], bandmap[vpr_bandmap,2]]


                # 2.0 we need to find these new k/q/Q in the 1s BZ:
                # res1 <- [Q+q, k+Q+q, k+Q]; res2 <- [Q+q, k-q, k+Q]
                # todo: actually, we can move Q+q out of this loop, since this is actually a constan for the given Q and q
                # kmap_dic = {'  %.6f    %.6f    %.6f' : [Q, k_acv, q, k_gkk], ...}
                # kmapout[x] = [Q, k_acv, q, k_gkk]
                Q_plus_q_point =  move_k_back_to_BZ_1(kmap[Q_kmap,0:3] + kmap[q_kmap,0:3])
                Q_plus_q_kmapout = kmap_dic['  %.6f    %.6f    %.6f'%(Q_plus_q_point[0], Q_plus_q_point[1], Q_plus_q_point[2])]
                k_minus_q_point = move_k_back_to_BZ_1(kmap[k_kmap,0:3] - kmap[q_kmap,0:3])
                k_minus_q_kmapout = kmap_dic['  %.6f    %.6f    %.6f'%(k_minus_q_point[0], k_minus_q_point[1], k_minus_q_point[2])]

                # 3.0 Calculation!
                # acvmat.shape = (nQ,nS,nk,nc,nv,2)
                # gkkmat.shape = (nq,nk,ni,nj,nmode)

                # (i) Acv_temp1
                # this is k'=k-q and Q'=Q+q for Acv_temp1:
                # it seems there is no need to find k_pr (think more...)
                # kpr_as_Q_plus_q_plus_k_acv = Q_plus_q_plus_k_kmapout[1] # 1 means the fifth column of kmap, which is k_acv
                Qpr_as_Q_plus_q_acv_index = Q_plus_q_kmapout[0]  # 0 means the fourth column of kmap, which is Q
                kpr_as_k_minus_q_acv_index = k_minus_q_kmapout[1] # 1 means the fifth column of kmap, which is k_acv

                Acv_temp1_second = np.complex(
                    acvmat[int(Qpr_as_Q_plus_q_acv_index), int(m_ex_acv_index), int(kpr_as_k_minus_q_acv_index), int(c_acv_index), int(v_acv_index)][0],
                    acvmat[int(Qpr_as_Q_plus_q_acv_index), int(m_ex_acv_index), int(kpr_as_k_minus_q_acv_index), int(c_acv_index), int(v_acv_index)][1]
                )
                Acv_temp1_conjugated_second = np.conjugate(Acv_temp1_second)

                # (ii) Acv_temp2
                # this is Q for Acv_temp2 (Q_acv_index has been found in previous step), so we can start to do Acv_temp2
                Acv_temp2_second = np.complex(
                    acvmat[int(Q_acv_index), int(n_ex_acv_index), int(k_acv_index), int(c_acv_index), int(vpr_acv_index)][0],
                    acvmat[int(Q_acv_index), int(n_ex_acv_index), int(k_acv_index), int(c_acv_index), int(vpr_acv_index)][1]
                )

                # (iii) gkk_temp
                # this is kpr = k + Q
                # kmapout[x] = [Q, k_acv, q, k_gkk]
                # kpr_as_k_minus_Q_gkk_index = k_plus_Q_kmapout[3] # 0 means the seventh column of kmap, which is k_gkk
                kpr_as_k_minus_q_gkk_index = k_minus_q_kmapout[3]
                gkk_temp_second = gkkmat[int(q_gkk_index), int(kpr_as_k_minus_q_gkk_index),int(vpr_gkk_index),int(v_gkk_index),int(v_ph_gkk)] # initial is c ->i; final is c' -> j

                # (iv) res_temp
                second_res = second_res + Acv_temp1_conjugated_second * Acv_temp2_second * gkk_temp_second

    res = res + first_res + second_res