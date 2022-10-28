# (1) readwhat_map = {1:"Qpr",2:"kpt_acv",3:"qpt",4:'kpt_gkk'}

# (2) kkqQmap.dat: This is a map from kgrid to index of [gkk] and [Acv]
# grid_1 grid_2 grid_3 Q k_acv q k_gkk
# kx ky kz Q_acv, k_acv, q_agkk, k_acv

# (3) kmap_dic = {'  %.5f    %.5f    %.5f' : [Q_acv, k_acv, q_acv, k_gkk], ...}

# (4) kmapout = [Q, k_acv, q, k_gkk] given ['  %.5f    %.5f    %.5f'] in kmap_dic

# (5) Qq_DIC = {'  %.5f    %.5f    %.5f' : Qq_fine}, where Qq_fine is index of interpolated index in gqQ_interpolated(q), omega(q) and OMEGA(Q)

