import numpy as np
import h5py as h5


class eqp():
    """
    These object decompose eqp.dat into data_LDA, data_GW, klist and spin_list
    """
    def __init__(self,fname):
        print('Reading eqp')
        self.fname = fname
        self.read_eqp()
        # These variables are initialized
        # (1) nbnd, nk
        # (2) data_GW, data_LDA -> (nk, nbnd)
        # (3) band_index, spin ->(nband,);
        # (4) klist -> (nk,)

        # self.write()

    def read_eqp(self):
        f = open(self.fname, 'r')
        lines = f.readlines()
        f.close()

        self.nbnd = int(lines[0].split()[-1])
        self.nk = int(len(lines) / (self.nbnd + 1))

        print('number of bands:', self.nbnd)
        print('number of kpoints:', self.nk)

        self.data_GW = np.zeros((self.nk, self.nbnd))
        self.data_LDA = np.zeros((self.nk, self.nbnd))
        self.band_index = np.zeros((self.nk, self.nbnd),dtype=int)
        self.spin_index = np.zeros((self.nk, self.nbnd),dtype=int)
        self.klist = []

        # get
        line = 0
        for i in range(self.nk):
            temp_k = lines[line]
            self.klist.append("  ".join(temp_k.split()[:3]))
            # print(" ".join(temp_k.split()[:3]))
            line += 1
            for j in range(self.nbnd):
                self.band_index[i,j] = int(lines[line].split()[1])
                self.spin_index[i,j] = int(lines[line].split()[0])
                self.data_LDA[i,j] = lines[line].split()[-2]
                self.data_GW[i,j] = lines[line].split()[-1]
                line += 1
        self.data_LDA = np.around(self.data_LDA,9)


    def write(self):
        f_new = open('eqp_new.dat','w')
        line = 0
        for i in range(self.nk):
            f_new.write('  '+self.klist[i]+'      %s'%self.nbnd+'\n')
            line += 1
            for j in range(self.nbnd):
                f_new.write('       %s     %s    %.9f    %.9f\n' % (self.spin_index[i,j], self.band_index[i,j], self.data_LDA[i,j], self.data_GW[i,j]))
                # print('%s %s %s %s' % (self.spin_index[j], self.band_index[j], self.data_LDA[i,j], self.data_GW[i,j]))
                line += 1

    def plot_eig(self):
        f = open('band.dat', 'w')
        for i in range(self.nbnd):
            for j in range(self.nk):
                f.write("%s %s %s\n" % (j + 1, self.data_LDA[j, i], self.data_GW[j, i]))
            f.write('\n')

        f.close()


if __name__ == "__main__":
    fname = "eqp_3QL_18181.dat"
    eqp_3QL = eqp(fname=fname)

    correction = eqp_3QL.data_GW - eqp_3QL.data_LDA
    correction_band_range = [max(eqp_3QL.band_index[:, 0]), min(eqp_3QL.band_index[:, -1])]


    band_shift = 6*78 # 5QL

    mf_5QL = np.loadtxt('bands_9QL_18181.dat')
    nk = len(eqp_3QL.klist)
    nb = mf_5QL.shape[0] // nk
    LDA_5QL = (
    (mf_5QL[:, 1].reshape(nb, nk))[correction_band_range[0] - 1 + band_shift: correction_band_range[1] + band_shift,
    :]).T



    eqp_3QL.data_LDA = LDA_5QL
    eqp_3QL.data_GW  = LDA_5QL + correction
    eqp_3QL.band_index = eqp_3QL.band_index + band_shift


    eqp_3QL.write()



    # -----------------------------------------------------------------------------------------
    # GW c-v
    # correction = eqp_3QL.data_GW - eqp_3QL.data_LDA
    # correction_band_range = [max(eqp_3QL.band_index[:,0]), min(eqp_3QL.band_index[:,-1])]
    #
    #
    # # 5QL
    # shift_band = 2 * 78
    # mf_5QL = np.loadtxt('bands_5QL.dat')
    # nk =  len(eqp_3QL.klist)
    # nb = mf_5QL.shape[0] // nk
    # mf_5QL = ((mf_5QL[:,1].reshape(nb,nk))[correction_band_range[0]-1+shift_band : correction_band_range[1]+shift_band,:]).T
    # mf_5QL_corrected = mf_5QL + correction
    # np.savetxt('bands_5QL_corrected.dat',mf_5QL_corrected)
    #
    # # 7QL
    # shift_band = 4 * 78
    # mf_7QL = np.loadtxt('bands_7QL.dat')
    # nk =  len(eqp_3QL.klist)
    # nb = mf_7QL.shape[0] // nk
    # mf_7QL = ((mf_7QL[:,1].reshape(nb,nk))[correction_band_range[0]-1+shift_band : correction_band_range[1]+shift_band,:]).T
    # mf_7QL_corrected = mf_7QL + correction
    # np.savetxt('bands_7QL_corrected.dat',mf_7QL_corrected)
    #
    # # 9QL
    # shift_band = 6 * 78
    # mf_9QL = np.loadtxt('bands_9QL.dat')
    # nk =  len(eqp_3QL.klist)
    # nb = mf_9QL.shape[0] // nk
    # mf_9QL = ((mf_9QL[:,1].reshape(nb,nk))[correction_band_range[0]-1+shift_band : correction_band_range[1]+shift_band,:]).T
    # mf_9QL_corrected = mf_9QL + correction
    # np.savetxt('bands_9QL_corrected.dat',mf_9QL_corrected)

    # mf_5QL = f['data'][0,:,correction_band_range[0]-1:correction_band_range[1]-1]


    #-----------------------------------------------------------------------------------------
    # nvalence = 10
    # scissor_shift = 0.5 # [eV]

    # do cv apart
    # eqp1.data_LDA[:,nvalence:] = eqp1.data_LDA[:,nvalence:] + scissor_shift
    #  eqp1.data_GW[:,nvalence:]  = eqp1.data_GW[:,nvalence:] + scissor_shift

    # eqp1.write()
    # eqp1.plot_eig()

    # for i in range(eqp1.band_index.shape[0]):
    #     if eqp1.band_index[i,0] == 235:
    #         print(eqp1.klist[i])





# import numpy as np
#
# eqp_name = 'eqp.dat'
# min_band = 230
# max_band = 240
#
# f = open(eqp_name,'r')
# lines = f.readlines()
# f.close()
#
# nbnd = int(lines[0].split()[-1])
# nk = int(len(lines)/(nbnd+1))
#
# print('number of bands:', nbnd)
# print('number of kpoints:', nk)
#
# data_GW = np.zeros((nk,max_band - min_band + 1))
# data_LDA = np.zeros((nk,max_band - min_band + 1))
#
# line = 0
# for i in range(nk):
#     line += 1
#     omit = 0
#     for j in range(nbnd):
#         if min_band <= int(lines[line].split()[1]) <= max_band:
#             data_LDA[i, j-omit] = lines[line].split()[-2]
#             data_GW[i, j-omit]   = lines[line].split()[-1]
#         else:
#             omit += 1
#         line += 1
#
# data_GW.sort(axis=1)
# data_LDA.sort(axis=1)

# f = open('band.dat','w')
# for i in range(max_band - min_band + 1):
#     for j in range(nk):
#         f.write("%s %s %s\n"%(j+1, data_LDA[j,i], data_GW[j,i]))
#     f.write('\n')
#
# f.close()