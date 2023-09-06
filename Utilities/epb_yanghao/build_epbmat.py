from epbfile import epbfile
import sys

fname = 'epbmat_merge.h5'
# 60 bands
ifmax = 40
nv = 40
nc = 20

epb = epbfile(fname=fname)
epb.diagonalize_dynmat()
epb.write_hdf5_qbyq(ifmax, nv, nc, 'epbmat_mode_q0.h5')


#epb.get_gkk_mode()
#epb.reorder_bands(ifmax, 4, 4)
#epb.reorder_kq()
#epb.write_hdf5('epbmat_mode.h5')
