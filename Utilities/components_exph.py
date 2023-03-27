import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

k_select = -1 # select the index of k (if=-1 -> all kpoints will be included)
scale = 10E2
e_min = 0
e_high = 5


f = h5.File('eigenvectors.h5','r')
evals = f['exciton_data/eigenvalues'][()]
evc = f['exciton_data/eigenvectors'][()]
evc = evc[0,:,:,:,:,0,:]
(nS, nk, nc, nv, _) = evc.shape


plt.figure(figsize=(16,8))
plt.subplots_adjust(left=0.22, right=0.94, top=0.96, bottom=0.22 , wspace=0, hspace=0)
colors=['r','g','b','y']

if k_select == -1:
    pass
else:
    evc = evc[:,[k_select],:,:,:]

for iN_S in range(nS):
    if e_min<evals[iN_S]<e_high:
        temp_contrib_cv = np.sum(abs(evc[iN_S,:,:,:,0]+evc[iN_S,:,:,:,1]*1j)**2, axis=0)#(ic,iv)

        temp_contrib_v = np.sum(temp_contrib_cv, axis=0) #(iv)
        temp_contrib_c = np.sum(temp_contrib_cv, axis=1) #(ic) #todo: finish the rest
        # plt.scatter(np.ones(nc) * evals[iN_S], np.array([np.arange(2,nc+1,2),np.arange(2,nc+1,2)]).T.reshape(nc), color='b', s=temp_contrib_c * scale)
        # plt.scatter(np.ones(nv) * evals[iN_S], np.array([np.arange(-2,-nv-1,-2),np.arange(-2,-nv-1,-2)]).T.reshape(nv), color='r', s=temp_contrib_v * scale)
        plt.scatter(np.ones(nc) * evals[iN_S], np.arange(1,  nc+1, 1), color='b', s=temp_contrib_c * scale)
        plt.scatter(np.ones(nv) * evals[iN_S], np.arange(-1,-nv-1,-1), color='r', s=temp_contrib_v * scale)
        print ("N_S:", iN_S)
    else:
        pass

plt.yticks([-2,-1,0,1,2,3,4,5,6,7,8],('VB-2','VB-1','Fermi Level','CB-1','CB-2','CB-3','CB-4','CB-5','CB-6','CB-7','CB-8'))

# plt.yticks([-2,-1,0,1,2],('VB-9','VB-8','VB-7','VB-6','VB-5','VB-4','VB-3','VB-2','VB-1','Fermi Level',
#                                         'CB-1','CB-2','CB-3','CB-4','CB-5','CB-6','CB-7'))
plt.ylabel('Contributing Band')
plt.xlabel('Exciton Energy (eV)')

plt.savefig('evec_component'+'.png',dpi=400)
