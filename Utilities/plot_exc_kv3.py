import h5py
import numpy as np

evec_filename = 'eigenvectors.h5'
nS_start = 881
nS_end = 882
nc = 4
nv = 4
weight_by_os = True




f = h5py.File(evec_filename,'r')
kpt = f['exciton_header/kpoints/kpts']
num_k = kpt.shape[0]
bvec = f['mf_header/crystal/bvec'][()]
osc = np.loadtxt('eigenvalues.dat')[:,1]
energy = np.loadtxt('eigenvalues.dat')[:,0]
# k = open('kgrid.out')
# k_info = k.readlines()
# k.close()
# del k_info[0:2]
State = ((energy <0.14)&(energy>0.11)) & (osc<0.1)

# def AcvS_reading(S,k,c,v): # all arguments follow fortran convention, ex: S=1 represents 1st exciton state
#     imagi=1j
#     AcvS_Re = f['exciton_data/eigenvectors'][0,S-1,k-1,c-1,v-1,0,0]
#     AcvS_Im = f['exciton_data/eigenvectors'][0,S-1,k-1,c-1,v-1,0,1]
#     return AcvS_Re + AcvS_Im * imagi

def AcvS_plot_data(State,c_n,v_n): # prepare |AcvS(K)|^2 dataFrame: [k1 k2 k3 |AcvS(k)|^2]
    AcvS_temp_matrix = f['exciton_data/eigenvectors'][0,State,:,0:c_n,0:v_n,0,:]
    print('shape of AcvS_temp_matrix:',AcvS_temp_matrix.shape)
    osc_temp = osc[State]

    if weight_by_os:
        AcvS_temp_matrix = AcvS_temp_matrix * osc_temp[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]

    exc = open('exc_wfn_filter.dat','w')
    for i in range(num_k):
        # AcvS_2 = 0
        k_inv = bvec[0]*kpt[i][0] + bvec[1]*kpt[i][1] + bvec[2]*kpt[i][2]
        # for nS in range(S_start,S_end+1):
        #     for j in range(c_n):
        #         for k in range(v_n):
        #             AcvS_2 = AcvS_2 + abs(AcvS(nS,i+1,j+1,k+1))
        AcvS_2 = np.abs(AcvS_temp_matrix[:,i,:,:,0]+1j*AcvS_temp_matrix[:,i,:,:,1]).sum(axis=(0,1,2))
        exc.write(str(k_inv[0])+' '+str(k_inv[1])+' '+str(k_inv[2])+' '+str(AcvS_2)+'\n')
        print("kpt:",i,'/',num_k,'  A(k):',AcvS_2)
    exc.close()

if __name__ == '__main__':
    AcvS_plot_data(State,nc,nv)
    f.close()
