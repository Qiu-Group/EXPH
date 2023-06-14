import numpy as np
import h5py as h5

def band_summary(v_start_gkk=11, c_end_gkk=14):
    """
    todo: write some function to find v_start_gkk and c_end_gkk
    """
    print('band check start')
    try:
        acv = h5.File("Acv.h5", 'r')
        gkk = h5.File("gkk.h5", 'r')
    except:
        raise Exception("Acv.h5 and gkk.h5 can not be opened")
    occ = int(np.sum(acv['mf_header/kpoints/occ'][0][0]))
    nc_acv = acv['exciton_header/params/nc'][()]
    nv_acv = acv['exciton_header/params/nv'][()]
    # v_start_gkk = 11  # tododone: write  some function to find this
    # c_end_gkk = 14  # tododone: write  some function to find this
    if v_start_gkk > occ or c_end_gkk <= occ:
        raise Exception("There is no conduction or valence band in gkk")
    # i) get boundary for c and v
    acv_band_fortran_order = list(
        range(occ - nv_acv + 1, occ + nc_acv + 1))  # Fortran order means band index start with 1
    gkk_band_fortran_order = list(range(v_start_gkk, c_end_gkk + 1))
    band_intersection = list(set(acv_band_fortran_order).intersection((set(gkk_band_fortran_order))))
    band_intersection.sort() # Bowen Hou 04/11/2023 it seems that sometimes the intersection is not rightly ordered.
    f = open('bandmap.dat', 'w')
    f.write("# band_quantum_number  acv   gkk   occ: %s\n" % occ)

    for band_index in band_intersection:  # find conduction and valence band
        if band_index < occ + 0.5:
            f.write('%s  %s  %s\n' % (band_index, occ - band_index, band_index - v_start_gkk))
        else:
            f.write("%s  %s  %s\n" % (band_index, band_index - occ - 1, band_index - v_start_gkk))

    f.close()
    print("[band check]: pass")
    print("[bandmap]: created")