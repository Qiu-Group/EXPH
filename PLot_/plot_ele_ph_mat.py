from IO.IO_common import write_loop
from Common.common import frac2carte
from Common.h5_status import check_h5_tree
from ELPH.EX_PH_mat import gqQ, gqQ_inteqp_q_nopara
from IO.IO_common import read_kmap, read_lattice
from IO.IO_acv import read_Acv
from IO.IO_gkk import read_gkk, read_omega, create_gkkh5
from ELPH.EX_PH_inteqp import OMEGA_inteqp_Q, omega_inteqp_q
from IO.IO_common import read_bandmap, read_kmap,construct_kmap
from Common.progress import ProgressBar
from IO.IO_common import read_kmap, read_lattice
from ELPH.EX_PH_inteqp import dispersion_inteqp_complete
from Common.inteqp import interqp_2D
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import sys
 # matplotlib.pyplot.colorbar
import numpy as np
import h5py as h5

path = '../'
interposize = 120
k_kmap_index = 3
initial_state_gkk_indsex = 2
final_state_gkk_index = [2]
mode_index = [3,4,5,6,7,8]

kmap = read_kmap(path=path)  # load kmap matrix
q_inteqp_no = int(np.sqrt(kmap.shape[0]))
# findal state: all bands
# loop over nq

# elphmat.shape = (nq,nk,ni,nj,nmode)
elph_mat = read_gkk(path=path)
[qxx_new, qyy_new, omega_res] = omega_inteqp_q(interpo_size=interposize, new_q_out=True, path=path)

interposize = interposize + 1
q_gkk_index_list = list(map(int, kmap[:, 5]))


res_elph_mat_temp0 = elph_mat[:,k_kmap_index,initial_state_gkk_indsex,:,:]
res_elph_mat_temp1 = res_elph_mat_temp0[:,final_state_gkk_index,:]
res_elph_mat_temp2 = res_elph_mat_temp1[:,:,mode_index]

res_elph_mat_res_nointeqp_temp0 = np.sum(np.sum(res_elph_mat_temp2,axis=1), axis=1)[np.array(q_gkk_index_list)].reshape((q_inteqp_no,q_inteqp_no))

res_elph_mat_res_nointeqp_temp1 = dispersion_inteqp_complete(res_elph_mat_res_nointeqp_temp0)
res_elph_mat_res_nointeqp_temp2= interqp_2D(res_elph_mat_res_nointeqp_temp1, interpo_size=interposize, kind='linear')[:interposize - 1,
                  :interposize - 1]

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(qxx_new, qyy_new, res_elph_mat_res_nointeqp_temp2 ,cmap=cm.cool)
plt.contourf(qxx_new, qyy_new, res_elph_mat_res_nointeqp_temp2)
cbar = plt.colorbar()
cbar.solids.set_edgecolor("face")
plt.show()