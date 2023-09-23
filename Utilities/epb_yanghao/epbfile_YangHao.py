# Adopt from Gab's GkkFile
import os
import numpy as np
from numpy import zeros, einsum
import h5py as h5
from constants import tol5, tol6, me_amu, kb_HaK, Ry2Thz, Ry2meV, Ry2cm1


class epbfile():

    asr = True

    def __init__(self, fname=None, read=True):

        self.fname = fname

        if read and self.fname:
            self.read_h5()
    
    def read_h5(self, fname=None):
        """Open the epb.h5 file and read it."""
        fname = fname if fname else self.fname

        if not os.path.exists(fname):
            raise OSError('File not found: {}'.format(fname))

        with h5.File(fname, 'r') as root:

            self.nk     = root.attrs.get('nk')
            # my_nk = nks
            self.my_nk  = root.attrs.get('my_nk')
            self.nat    = root.attrs.get('nat')
            self.nbnd   = root.attrs.get('nbnd')
            self.nmodes = root.attrs.get('nmodes')
            self.nq     = root.attrs.get('nq')

            # nat
            self.ityp   = root.get('ityp')[()]
            self.ityp = self.ityp - 1 
            # python count from 0

            self.amass  = root.get('amass')[()]
            # nks, nbnd
            self.et     = root.get('et')[()]
            # 3, 3
            self.bvec   = root.get('bvec')[()]
            # 3, 3
            self.epsi   = root.get('epsi')[()]
            # nks, 3
            tmp         = root.get('my_xk')[()]
            self.my_xk  = self.fold_kpts(self.crys_to_cart(tmp, -1)) 
            # nq, 3
            tmp         = root.get('xqc')[()]
            self.xqc    = self.fold_kpts(self.crys_to_cart(tmp, -1))
            # natom, 3, 3
            self.zstar  = root.get('zstar')[()]
            # nq, nmodes, nmodes
            tmp = root.get('dynq')[()]
            tmp = np.reshape(tmp, [self.nq, self.nmodes, self.nmodes, 2])
            self.dynq = tmp[...,0] + 1j*tmp[...,1]
            self.dynq = self.dynq.transpose([0,2,1])

            # Control order of epmatq
            self.nk_first = root.get('nk_first')[()]
            ## nq, nmodes, nks, nbnd, nbnd
            # Memory issue # Bowen Hou: since you just import all elphmat to memory..
            tmp = root.get('epmatq')[()]
            tmp = np.reshape(tmp, [self.nq, self.nmodes, self.my_nk, self.nbnd, self.nbnd, 2])
            self.GKK = tmp[...,0] + 1j*tmp[...,1]

        self.omega = np.zeros([self.nq, self.nmodes])
        self.eigvect = np.zeros([self.nq, self.nmodes, self.nmodes], dtype=complex)

        self.dyn_diagonalized = False
        self.GKKmode_done = False
        self.reordered = False
        self.kreorder = False

        self.print_variables()


    def diagonalize_dynmat(self, dont_scale=False):

        for iqpt in range(self.nq):
           self.omega[iqpt], self.eigvect[iqpt] = self.diagonalize_dynmat_q(iqpt, dont_scale)

        self.dyn_diagonalized = True

        for iqpt in range(self.nq):
           print( ' Phonon frequency at qpt:', iqpt)
           for imode in range(self.nmodes):
              #print( '   mode, freq in Thz:', self.omega[iqpt,imode]*Ry2Thz )
              print( '         freq in cm-1:', self.omega[iqpt,imode]*Ry2cm1)

        # Set w(q) = w(-q)
        print( ' Impose w(q) = w(-q)')
        checked = []
        for iqpt, qpt in enumerate(self.xqc):
          if iqpt not in checked:
             rqpt = self.fold_kpts(np.array([-qpt]))
             irqpt = np.argmin(np.linalg.norm(rqpt-self.xqc, axis=1))
             self.omega[irqpt,:] = self.omega[iqpt,:]
             checked.append(iqpt)
             checked.append(irqpt)


    def diagonalize_dynmat_q(self, iqpt, dont_scale=False):
        """
        diagonalize dynamical matrix at a single q
        check rotate_eigenm.f90
        """
        dynp = np.zeros(self.dynq[iqpt].shape, dtype=complex)

        for i in range(self.nat):
          for j in range(self.nat):
             massfac = 1./ np.sqrt( self.amass[self.ityp[i]] * self.amass[self.ityp[j]])
             dynp[3*i:3*(i+1), 3*j:3*(j+1)] =\
                 self.dynq[iqpt, 3*i:3*(i+1), 3*j:3*(j+1)] * massfac

        dyn1 = (dynp + dynp.transpose().conjugate())/2.0
        evals, eigvecs = np.linalg.eigh(dyn1)

        # Set imaginary to 0.0
        for i, eig in enumerate(evals):
          if eig < 0.0:
            evals[i] = 0.0

        # Scale the eigenvectors
        if not dont_scale:
          for i in range(self.nat):
            for dir1 in range(3):
              ipert = i*3 + dir1
              eigvecs[ipert,:] = eigvecs[ipert,:] * np.sqrt( 1./ self.amass[self.ityp[i]])

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

    def get_gkk_mode(self):
        """
        Rotate GKK to mode representation
        GKK is in Cartesian representation from EPW
        check rotate_epmat.f90

        We further divided by zero-point displacement
        to get g_mnv(k,q)

        """
        if not self.dyn_diagonalized:
           self.diagonalize_dynmat()

        # GKK: nq, nmodes, nks, nbnd, nbnd
        self.GKKmode = np.einsum('qikst,qij->qjkst', self.GKK, self.eigvect)

        # GKK needs to be multiplied by zero-point displacement
        # l_qv = ( hbar / (2*M0*w_qv) )**0.5
        # to get g_mnv(k,q) in the effective hamiltonian

        for iq in range(self.nq):
          for i in range(self.nmodes):
            if self.omega[iq,i] > 1e-5:
              self.GKKmode[iq,i,...] =\
                      self.GKKmode[iq,i,...] / np.sqrt(2*self.omega[iq,i])
            else:
              self.GKKmode[iq,i,...] = 0.0
         
        self.GKKmode_done = True

        return


    def get_gkk_squared(self):
        """
        Get squared values of gkk

        The factor sqrt(hbar/2/M/omega) is considered in get_gkk_mode
        M is hidden in the normal modes

        """
        if not self.GKKmode_done:
           self.get_gkk_mode()

        # nq, nmode, nks, nbnd, nbnd
        gkk2 = np.einsum('qjkst,qjkst->qjkst',\
                          self.GKKmode.conjugate(), self.GKKmode)
        # Symmetrize ?

        return gkk2


    def crys_to_cart(self, kpts_in, mode):
        """
        Convert between Cartesian and crystal coordinates

        """
        kpts_out = np.zeros(kpts_in.shape)

        if mode < 0:
           # Cart to Cryst
           binv = np.linalg.inv(self.bvec)
           for ik, kpt in enumerate(kpts_in):
             kpts_out[ik] = np.dot(kpt, binv)
        else:
           # Cryst to Cart
           for ik, kpt in enumerate(kpts_in):
             kpts_out[ik] = np.dot(kpt, self.bvec)

        return kpts_out


    def fold_kpts(self, kpts_in):
        """
        kpts in the range of [0,1)

        """
        kpts_out = np.zeros(kpts_in.shape)

        for ik, kpt in enumerate(kpts_in):
           for i in range(3):
             while kpt[i] < -1e-10:
               kpt[i] = kpt[i] + 1.0
             while kpt[i] >= 1-1e-10:
               kpt[i] = kpt[i] - 1.0

           kpts_out[ik] = kpt

        return kpts_out

    def reorder_bandsq(self, gmode, ifmax, nvb_u, ncb_u):
        """
        Reorder bands at a single q point,
        since BerkeleyGW counts bands from Fermi level

        Input:
            ifmax: valence band top
            nvb_u, ncb_u: bands number we want to use

        """
        nvb_q = ifmax - nvb_u
        ncb_q = ifmax + ncb_u
        if nvb_q < 0 or ncb_q > self.nbnd:
           raise Exception('  epb matrix does not have enough bands' )

        nbnd = nvb_u + ncb_u
        gmode_out = np.zeros([self.nmodes,self.nk,nbnd,nbnd], dtype=complex)

        idx = np.append(np.arange(ifmax-1,ifmax-1-nvb_u,-1),\
                        np.arange(ifmax, ifmax+ncb_u))
        order = np.ix_(range(self.nmodes),range(self.nk),idx,idx)
        gmode_out = gmode[order]

        return gmode_out

    def reorder_bands(self, ifmax, nvb_u, ncb_u):
        """
        Reorder bands,
        since BerkeleyGW counts bands from Fermi level

        Input:
            ifmax: valence band top
            nvb_u, ncb_u: bands number we want to use

        """
        if not self.GKKmode_done:
            raise Exception('GKK mode representation is not computed.')

        self.nvb = nvb_u
        self.ncb = ncb_u
        self.ifmax = ifmax

        nbnd = self.nvb+self.ncb

        Gkktmp = np.zeros([self.nq,self.nmodes,self.nk,nbnd,nbnd], dtype=complex)

        for ik in range(self.nk):

           nvbtop = ifmax
           nvb_q = nvbtop - nvb_u
           ncb_q = nvbtop + ncb_u

           if nvb_q < 0 or ncb_q > self.nbnd:
              raise Exception('  epb matrix does not have enough bands' )

           idx = np.append(np.arange(nvbtop-1,nvbtop-1-nvb_u,-1),\
                           np.arange(nvbtop, ncb_u+nvbtop))
           order = np.ix_(range(self.nq),range(self.nmodes),range(ik,ik+1),\
                          idx,idx)
           tmp = self.GKKmode[order]
           Gkktmp[:,:,ik,:,:] = tmp.reshape([self.nq,self.nmodes,nbnd,nbnd])

        self.GKKmode = np.array(Gkktmp[:])

        self.reordered = True
        print( '  New shape of GKK', self.GKKmode.shape)

        return

    def reorder_kq(self):
        """
        g(k,q)_mn = < m k+q | Vq | n k >

        We relable it as

        g'(k',k)_mn = < m k' | V(k'-k) | n k>

        """
        if self.kreorder:
            print( " Reordering has been done")
            return

        tmp = np.zeros(self.GKKmode.shape, dtype=complex)

        for ik, kpt in enumerate(self.my_xk):
          for iq, qpt in enumerate(self.xqc):
            kpq = self.fold_kpts(np.array([kpt+qpt]))
            ikpq = np.argmin(np.linalg.norm(kpq-self.my_xk, axis=1))
            tmp[ikpq,:,ik,:,:] = self.GKKmode[iq,:,ik,:,:]

        self.GKKmode = None
        self.GKKmode = np.array(tmp[:])

        self.kreorder = True

        nb = self.GKKmode.shape[-1]
        # Symmetrize. To ensure detailed balance
        for ik in range(self.nk):
          for ikp in range(ik):
            for mu in range(self.nmodes):
              for nn in range(nb):
                for mm in range(nn+1):
                   self.GKKmode[ikp,mu,ik,mm,nn] = self.GKKmode[ik,mu,ikp,nn,mm].conjugate()

        return

    def print_variables(self):

        print( ' Data in the file:', self.fname)
        if self.nk_first:
           print('\n  Warning!')
           print('  epmatq is ordered as [nk, nq, nmode, nb, nb]')
        print( '   Total number of kpoints = ', self.nk)
        print( '   Local number of kpoints = ', self.my_nk)
        for kpt in self.my_xk:
            print( kpt)

        print( '   Number of bands = ', self.nbnd)
        print( '   Number of modes = ', self.nmodes)
        print( '   Number of qpoints= ', self.nq)

        for qpt in self.xqc:
            print( qpt)

    def write_phonons(self, fout='phonon.h5'):
        """
        write phonon frequency and eigenvectors

        """
        if not self.dyn_diagonalized:
          self.diagonalize_dynmat()

        with h5.File(fout, 'w') as f:
          header = f.create_group("header")
          header.create_dataset("nq", data=self.nq)
          header.create_dataset("qpts", data=self.xqc)
          header.create_dataset("nmodes", data=self.nmodes)
          f.create_dataset("omega", data=self.omega)
          f.create_dataset("eigvect", data=self.eigvect)

        return
    #
    def write_hdf5_qbyq(self, ifmax, nvb_u, ncb_u, fout='epbmat_mode.h5'):
        """
        write phonon frequency and epbmat
        q-point by q-point to reduce memory

        """
        f_in = self.fname
        self.nvb = nvb_u
        self.ncb = ncb_u
        self.ifmax = ifmax
        nbu = nvb_u + ncb_u

        if not self.dyn_diagonalized:
          self.diagonalize_dynmat()

        with h5.File(fout, 'w') as f:
          header = f.create_group("header")
          header.create_dataset("nfk", data=self.nk)
          header.create_dataset("nk", data=self.my_nk)
          header.create_dataset("nq", data=self.nq)
          header.create_dataset("kpts", data=self.my_xk)
          header.create_dataset("qpts", data=self.xqc)
          header.create_dataset("nmodes", data=self.nmodes)
          header.create_dataset("nbnd", data=self.nbnd)

          header.create_dataset("nvb", data=self.nvb)
          header.create_dataset("ncb", data=self.ncb)
          header.create_dataset("ifmax", data=self.ifmax)
          f.create_dataset("omega", data=self.omega)
          f.create_dataset("GKK", (self.nq,self.nmodes,self.nk,nbu,nbu),\
                            dtype=np.complex)
          # write GKK
          with h5.File(f_in, 'r') as f2:
             for iq in range(self.nq):
               print('\n  Computing gkk at iq: %d'% iq)
               if self.nk_first:
                  tmp = f2.get('epmatq')[:,iq,...]
                  tmp = tmp.transpose([1,0,2,3])
               else:
                  tmp = f2.get('epmatq')[iq]
               tmp = np.reshape(tmp, [self.nmodes, self.my_nk, self.nbnd, self.nbnd, 2])
               gkk = tmp[...,0] + 1j*tmp[...,1]
               gmode = self.get_gkk_modeq(iq, gkk)
               gmode_reduced = self.reorder_bandsq(gmode, ifmax, nvb_u, ncb_u)
               f['GKK'][iq,...] = gmode_reduced

          return
    #
    # def write_hdf5(self, fname='epbmat_mode.h5'):
    #     """
    #     write phonon frequency and epbmat
    #     Reorder qpts, kpts ?
    #
    #     """
    #     with h5.File(fname, 'w') as f:
    #
    #       header = f.create_group("header")
    #       header.create_dataset("nfk", data=self.nk)
    #       header.create_dataset("nk", data=self.my_nk)
    #       header.create_dataset("nq", data=self.nq)
    #       header.create_dataset("kpts", data=self.my_xk)
    #       header.create_dataset("qpts", data=self.xqc)
    #       header.create_dataset("nmodes", data=self.nmodes)
    #       header.create_dataset("nbnd", data=self.nbnd)
    #
    #       if self.reordered:
    #          header.create_dataset("nvb", data=self.nvb)
    #          header.create_dataset("ncb", data=self.ncb)
    #          header.create_dataset("ifmax", data=self.ifmax)
    #
    #
    #       f.create_dataset("omega", data=self.omega)
    #       f.create_dataset("GKK", data=self.GKKmode)


