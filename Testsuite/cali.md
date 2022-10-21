cali:

(1) EL_PH_mat:

res = gqQ(n_ex_acv_index=8, m_ex_acv_index=3, v_ph_gkk=2, Q_kmap=3, q_kmap=11,path='../',k_map_start_para=77,k_map_end_para=144)

res = 0.0159358508119-0.000852284450595j


(2) EX_PH_cat:

series:
[Exciton Scattering]: n= 2  Q= 15 T= 100
8463757989509.6123

parallel (1 task):
[Exciton Scattering]: n= 2  Q= 15 T= 100
res is 8.46375798951e+12

parallel efficiency benchmark:


| mpi task    | wall time (s)    | CPU time (s) |
| :---: |   :---:       | :---: |
| 1        | 197.743        | 196.203   |
| 2        | 124.767         | 120.906   |
| 4 | 83.884 | 82.344|
| 6| 66.071|  64.609|
| 8 | 53.952 |  48.594
