cali:

1. EL_PH_mat:
12 12 1
res = gqQ(n_ex_acv_index=8, m_ex_acv_index=3, v_ph_gkk=2, Q_kmap=3, q_kmap=11,path='../',k_map_start_para=77,k_map_end_para=144)

newest!! res = 0.0159358508119-0.000852284450595j deprecated (wrong)

4 4 1
(-0.00029862310477371133-0.00022746828154937592j) wrong
[(1.2586895416293405e-05+4.465232765946882e-05j)] !!

2. EX_PH_cat (w/o normalization):

(a) 4 4 1
193574848.4561666?? should be wrong
7839570641.3534737
newest!!!!  8876085809266.0039 (bug!, depracated) (skip omega=0, delta_function sigma=0.001):
121658087995.95515 wrong: since mat is wrong + -> -
[50631692479.397934] !!

(b) 12 12 1
8.46375798951e+15<<<
newest!!!!!!  2.42461752574e+16 (bug!, depracated)


Warning: "normalization" should not be included
3. EX_PH_Cat (w/ normalization)
(a): 4-4-1
193574848.4561666

(b): 12 12 1 !!!
res is 45145155828.4





parallel efficiency benchmark:



(a) Test on PC Windows

| mpi task | wall time (s) | CPU time (s) |
|:--------:|:-------------:|:------------:|
|    1     |    197.743    |   196.203    |
|    2     |    124.767    |   120.906    |
|    4     |    83.884     |    82.344    |
|    6     |    66.071     |    64.609    |
|    8     |    53.952     |    48.594    |

(b) Test on Frontera

| mpi task    | wall time (s)    | CPU time (s) |
| :---: |   :---:       | :---: |
|2 | 156.251 | 155.896|
|4 |77.328 | 77.139|
| 8 | 39.579| 39.215|
|16| 19.389 |19.276|
| 32 | 10.068|  9.885|
|40 | 8.068 | 8.057|
|48| 6.130  | 6.119|
|56| 6.382 | 6.148|
