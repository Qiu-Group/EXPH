import numpy as np
import h5py as h5
from Utilities.constants import Ry2cm1

class epbfile():
    def __init__(self, prefix="MoS2", read=True):
        self.fname = prefix + '_elph_1.h5'

        if read and self.fname:
            self.read_dynq_h5()
            self.read_epmaq_h5()

    def read_dynq_h5(self):

        self.f_dynq = h5.File(self.fname,'r')
        self.amass = self.f_dynq.get('atomic_masses')[()]
        self.dynq_real = self.f_dynq.get('dynq_real')[()]
        self.dynq_imag = self.f_dynq.get('dynq_imag')[()]

        self.dynq = self.dynq_real + 1j * self.dynq_imag
        self.dynq = self.dynq.transpose([0,2,1])


        self.nq = self.dynq.shape[0]
        self.nk = self.dynq.shape[0]
        self.nmodes = self.dynq.shape[1]
        self.nat = self.nmodes // 3
        self.nband = self.f_dynq['elph_cart_real'].shape[-1]

        self.omega = np.zeros([self.nq, self.nmodes])
        self.eigvect = np.zeros([self.nq, self.nmodes, self.nmodes], dtype=complex)

    def read_epmaq_h5(self):
        self.f_epmatq = h5.File('el_ph_cart.h5','r')


    def diagonalize_dynmat(self, dont_scale=False):
        for iqpt in range(self.nq):
            self.omega[iqpt], self.eigvect[iqpt] = self.diagonalize_dynmat_q(iqpt, dont_scale)

        for iqpt in range(self.nq):
           print( ' Phonon frequency at qpt:', iqpt)
           for imode in range(self.nmodes):
              #print( '   mode, freq in Thz:', self.omega[iqpt,imode]*Ry2Thz )
              print( '         freq in cm-1:', self.omega[iqpt,imode]*Ry2cm1)

    def diagonalize_dynmat_q(self, iqpt, dont_scale=False):
        dynp = np.zeros(self.dynq[iqpt].shape, dtype=complex)

        # TODO: divide by mass
        for i in range(self.nat):
          for j in range(self.nat):
             # massfac = 1./ np.sqrt( self.amass[self.ityp[i]] * self.amass[self.ityp[j]])
             dynp[3*i:3*(i+1), 3*j:3*(j+1)] = self.dynq[iqpt, 3*i:3*(i+1), 3*j:3*(j+1)]
                                             # * massfac

        dyn1 = (dynp + dynp.transpose().conjugate())/2.0
        evals, eigvecs = np.linalg.eigh(dyn1)

        # Set imaginary to 0.0
        for i, eig in enumerate(evals):
          if eig < 0.0:
            evals[i] = 0.0

        # TODO: Scale the eigenvectors
        # if not dont_scale:
        #   for i in range(self.nat):
        #     for dir1 in range(3):
        #       ipert = i*3 + dir1
        #       eigvecs[ipert,:] = eigvecs[ipert,:] * np.sqrt( 1./ self.amass[self.ityp[i]])

        omega = np.sqrt(evals)
        eigvect = np.array(eigvecs[:])

        return omega, eigvect

    def get_gkk_modeq(self, iq, gkk):
        """
        Rotate GKK to mode representation at a single q
        GKK is in Cartesian representation from EPW
        check rotate_epmat.f90

        We further divided by zero-point displacement
        to get g_mnv(k,q)

        """
        # GKK: nq, nmodes, nks, nbnd, nbnd
        gkkmodeq = np.einsum('ikst,ij->jkst', gkk, self.eigvect[iq])

        # GKK needs to be multiplied by zero-point displacement
        # l_qv = ( hbar / (2*M0*w_qv) )**0.5
        # to get g_mnv(k,q) in the effective hamiltonian

        for i in range(self.nmodes):
          if self.omega[iq,i] > 1e-5:
             gkkmodeq[i,...] =\
                    gkkmodeq[i,...] / np.sqrt(2*self.omega[iq,i])
          else:
             gkkmodeq[i,...] = 0.0

        return gkkmodeq

    def write_hdf5_qbyq(self):

        # memory issue
        with h5.File("elphmat_phase.h5",'w') as f:
            f.create_dataset("data", (self.nq, self.nk, self.nband, self.nband, self.nmodes, 2), dtype=np.float)
            # f.create_dataset("elph_nu_imag", (self.nq, self.nmodes, self.nk, self.nband, self.nband), dtype=np.float)

            for iq in range(self.nq):
                print("\n Computing gkk at iq: %d"%iq)
                gkk = self.f_epmatq.get('elph_cart_real')[iq] + 1j * self.f_epmatq.get('elph_cart_imag')[iq] # (nq,nmu,nk,ni,nj) ->  (nmu,nk,ni,nj)
                gmode = self.get_gkk_modeq(iq,gkk) # (nmu,nk,ni,nj)
                gmode = gmode.transpose([1,2,3,0]) #(nk,ni,nj,nu)
                f["data"][iq,:,:,:,:,0] = np.real(gmode)
                f["data"][iq,:,:,:,:,1] = np.imag(gmode)
        return


if __name__ == "__main__":
    epb = epbfile()
    epb.diagonalize_dynmat()
    epb.write_hdf5_qbyq()