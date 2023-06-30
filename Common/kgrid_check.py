import numpy as np
import h5py as h5
from Common.h5_status import check_h5_tree
from Common.common import equivalence_order, equivalence_no_order, move_k_back_to_BZ_1, is_sqr
from IO.IO_common import readkkqQ, read_kgrid_log, construct_symmetry_dic
import os


# i) check a_lattice and b_lattice
def lattice_check(h5path='./'):
    acv = h5.File(h5path + 'Acv.h5', 'r')
    gkk = h5.File(h5path + 'gkk.h5', 'r')
    lattice_tolerance = 5E-4
    a_lattice_bohr_gkk = gkk['epw_header/a_lattice_bohr'][()]
    b_lattice_bohr_gkk = gkk['epw_header/b_lattice_bohr'][()]
    a_lattice_bohr_acv = acv['mf_header/crystal/avec_bohr'][()]
    b_lattice_bohr_acv = acv['mf_header/crystal/bvec_bohr'][()]
    if equivalence_order(a_lattice_bohr_acv, a_lattice_bohr_gkk, tolerance=lattice_tolerance) and equivalence_order(
            b_lattice_bohr_acv, b_lattice_bohr_gkk, tolerance=lattice_tolerance):
        print("[lattice check]: pass")
    else:
        print("[lattice check]: not pass")
        raise Exception("lattice from Acv.h5 does not match with lattice from gkk.h5")
    acv.close()
    gkk.close()


# ii) generate standard reciprocal grid -> (kx, ky, kz) is a crystal coordinate, 0 <= ki <= 1
# Therefore, we need a function which could move all k-points outside the 1st BZ back to the 1st BZ by applying G0 vector
# Then check if all k-grid match


def firstGridCheck(h5path='./',acvs_save_path='./save/acvs.save'):
    """
    Just take the raw k-grid from h5 and test if the k-grid match
    :param h5path:
    :return: number of not Matching relation
    """
    kpt_tolerance = 5E-4

    # see if symmetry used for Qpt
    if os.path.isfile(acvs_save_path+'/kgrid.log'):
        print('[symmetry] found in first kgrid_check')
        uniform_grid_array, reduced_grid_array = read_kgrid_log(acvs_save_path)
        # get Qpt from full-grid!! since we have symm_dic, which could transfer index of full_grid to reduced_grid index
        Qpt_in_acv_frac = uniform_grid_array[:,0:3]
    else:
        print('[symmetry] NOT found in the first kgrid_check')
        # if symmetry is not used, just get Qpt from Acvs.h5
        Qpt_in_acv_frac = readkkqQ(1, h5path)

    kpt_in_acv_frac = readkkqQ(2, h5path)
    qpt_in_gkk_frac = readkkqQ(3, h5path)
    kpt_in_gkk_frac = readkkqQ(4, h5path)
    # kpt_acv, Qpt_acv, kpt_gkk and qpt_gkk should match!! A(c,v,k,k+Q+q,S,Q+q), g(k+Q,q) & g(k-q,q)
    # Otherwise, choose small grid to do further calculations
    # e.g.: if k:16 16 1 and q: 4 4 1, we should only keep 4 4 1 kpoints as qpoints
    notMatch = 0
    if not equivalence_no_order(Qpt_in_acv_frac, kpt_in_acv_frac, kpt_tolerance):
        notMatch += 1
        print("WARNING: kpt and Qpt in Acv.h5 does not match")
    if not equivalence_no_order(kpt_in_acv_frac, kpt_in_gkk_frac, kpt_tolerance):
        notMatch += 1
        print("WARNING: kpt in gkk.h5 and kpt in Acv.h5 does not match")
        notMatch += 1
    if not equivalence_no_order(kpt_in_acv_frac, qpt_in_gkk_frac, kpt_tolerance):
        notMatch += 1
        print("WARNING: qpt in gkk.h5 and kpt in Acv.h5 does not match")
    if not equivalence_no_order(Qpt_in_acv_frac, kpt_in_gkk_frac, kpt_tolerance):
        notMatch += 1
        print('WARNING: Qpt in Acv.h5 doesn\'t match with kpt in gkk.h5')
    if not equivalence_no_order(Qpt_in_acv_frac, qpt_in_gkk_frac, kpt_tolerance):
        notMatch += 1
        print('WARNING: Qpt in Acv.h5 doesn\'t match with qpt in gkk.h5')
    if not equivalence_no_order(kpt_in_gkk_frac, qpt_in_gkk_frac, kpt_tolerance):
        notMatch += 1
        print("WARNING: qpt in gkk.h5 and kpt in gkk.h5 does not match")

    if notMatch == 0:
        print("[first grid check]: pass")
        print("[kkqQmap]: created")
    else:
        print("[first grid check]: not pass")
        print("We find", notMatch, "set k-grid not matching")
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print("WARNING: this will lead phase not aligned!!!")
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print("Reduced K-grid seeking...")


    if notMatch == 0:
        kmap_generate(Qpt_in_acv_frac, kpt_in_acv_frac, qpt_in_gkk_frac, kpt_in_gkk_frac, readwhat=4, acvs_save_path=acvs_save_path)
    return notMatch


def kmap_generate(Q, k_acv, q, k_gkk, readwhat=4, acvs_save_path = './save/acvs.save'):
    """
    :param Q: Q-grid in acv
    :param k_acv: k-grid in acv
    :param q: q-grid in gkk
    :param k_gkk: k-grid in gkk
    :param readwhat: choose [readwhat] as standard basis for all k-grid (ussually, this is the least dense k-grid), otherwise, choose q-gkk as basis
    :return:
    """
    #h5path='./'
    # matched or non matched grid
    # Q =     readkkqQ(1, h5path)
    # k_acv = readkkqQ(2, h5path)
    # q =     readkkqQ(3, h5path)
    # k_gkk = readkkqQ(4, h5path)
    # get base grid as k-grid standard

    # see if symmetry used for Qpt
    if os.path.isfile(acvs_save_path+'/kgrid.log'):
        # routine check
        uniform_grid_array, reduced_grid_array = read_kgrid_log(acvs_save_path,mute=True)
        symm_dic = construct_symmetry_dic(acvs_save_path,mute=True)
        # if Q == uniform_grid_array[:,0:3]:
        #     pass
        # else:
        #     raise Exception(" Q != uniform_grid_array[:,0:3]")

        # generate:
        baseKgrid = [Q, k_acv, q, k_gkk][readwhat-1] # choose [readwhat] as kgrid basis
        print('[symmetry] used in kmap')
        f=open('kkqQmap.dat', 'w')
        f.write('#This is a map from kgrid to index of gkk and Acv\n')
        f.write('# grid_1 grid_2 grid_3 Q k_acv q k_gkk\n')
        # res_map = {}
        for j in range(baseKgrid.shape[0]):
            base_kpoint = baseKgrid[j]
            res = '  %.7f    %.7f    %.7f'%(base_kpoint[0], base_kpoint[1], base_kpoint[2])
            # res_map['%.5f %.5f %.5f'%(base_kpoint[0], base_kpoint[1], base_kpoint[2])] = []
            for i in range(4):
                if i == 0:
                    # this is Q, for symmetrcy!
                    todoKgrid = [Q, k_acv, q, k_gkk][i]
                    match = 0
                    for k in range(todoKgrid.shape[0]):
                        # Here todoKgrid is Qpt (full grid)
                        if equivalence_order(baseKgrid[j],todoKgrid[k]):
                            match = 1
                            # print()
                            key_temp = '  %.7f    %.7f    %.7f' % (todoKgrid[k][0], todoKgrid[k][1], todoKgrid[k][2])
                            Q_index = symm_dic[key_temp.replace('-', '')]
                            res = res + "    %s"%Q_index
                            continue
                    if match == 0:
                        res = res+"    -1"
                else:
                    # for k_acv, q and k_gkk, there is no symmetry applied
                    todoKgrid = [Q, k_acv, q, k_gkk][i]
                    match = 0
                    for k in range(todoKgrid.shape[0]):
                        if equivalence_order(baseKgrid[j],todoKgrid[k]):
                            match = 1
                            res = res + "    %s"%k
                            continue
                    if match == 0:
                        res = res+"    -1"

            f.write(res+'\n')
        f.close()

    else:

        # see if nQ, nk, nq and nk are sqrt number
        [nQ, nk_acv, nq, nk_gkk] = [Q.shape[0], k_acv.shape[0],
                                    q.shape[0], k_gkk.shape[0]]
        if is_sqr(nQ) and is_sqr(nk_acv) and is_sqr(nq) and is_sqr(nk_gkk):
            pass
        else:
            raise Exception("If symmetry is not applied, please use uniform k-grid!!")

        print('[symmetry] NOT used in kmap')
        baseKgrid = [Q, k_acv, q, k_gkk][readwhat-1] # choose [readwhat] as kgrid basis
        f=open('kkqQmap.dat', 'w')
        f.write('#This is a map from kgrid to index of gkk and Acv\n')
        f.write('# grid_1 grid_2 grid_3 Q k_acv q k_gkk\n')
        # res_map = {}
        for j in range(baseKgrid.shape[0]):
            base_kpoint = baseKgrid[j]
            res = '  %.7f    %.7f    %.7f'%(base_kpoint[0], base_kpoint[1], base_kpoint[2])
            # res_map['%.5f %.5f %.5f'%(base_kpoint[0], base_kpoint[1], base_kpoint[2])] = []
            for i in range(4):
                todoKgrid = [Q, k_acv, q, k_gkk][i]
                match = 0
                for k in range(todoKgrid.shape[0]):
                    if equivalence_order(baseKgrid[j],todoKgrid[k]):
                        match = 1
                        res = res + "    %s"%k
                        continue
                if match == 0:
                    res = res+"    -1"
            f.write(res+'\n')
        f.close()



    # else:
    #     raise Exception("choose a base for k-grid")

# todo: deprecate secondGridCheck!!
def secondGridCheck(h5path='./', acvs_save_path = './save/acvs.save'):
    # find the least dense as basis
    # if match (no -1 in f): good!
    # else(-1 in f): raise Exception
    # see if symmetry used for Qpt
    if os.path.isfile(acvs_save_path+'/kgrid.log'):
        print('[symmetry] found in first kgrid_check')
        uniform_grid_array, reduced_grid_array = read_kgrid_log(acvs_save_path,True)
        Qpt_in_acv_frac = uniform_grid_array[:,0:3]
    else:
        print('[symmetry] NOT found in the first kgrid_check')
        Qpt_in_acv_frac = readkkqQ(1, h5path)

    kpt_in_acv_frac = readkkqQ(2, h5path)
    qpt_in_gkk_frac = readkkqQ(3, h5path)
    kpt_in_gkk_frac = readkkqQ(4, h5path)
    [nQ, nk_acv, nq, nk_gkk] = [Qpt_in_acv_frac.shape[0], kpt_in_acv_frac.shape[0],
                                qpt_in_gkk_frac.shape[0], kpt_in_gkk_frac.shape[0]]

    # if is_sqr(nQ) and is_sqr(nk_acv)

    readwhat = [nQ, nk_acv, nq, nk_gkk].index(min([nQ, nk_acv, nq, nk_gkk])) + 1
    kmap_generate(Qpt_in_acv_frac,kpt_in_acv_frac,qpt_in_gkk_frac,kpt_in_gkk_frac,readwhat=readwhat,acvs_save_path=acvs_save_path)
    f = np.loadtxt("kkqQmap.dat")
    if -1 in f:
        print("[second grid check]: not pass")
        raise Exception("Q q k k grid not matching!!")
    else:
        print("[second grid check]: pass")
        print("[kkqQmap]: created")


def k_grid_summary(h5path='./', acvs_save_path ='./save/acvs.save'):
    """
    :param h5path:
    :return: grid_match_result and kkQq map
    """
    # lattice check
    lattice_check(h5path)
    # 1st k-grid check
    notMatchFirstCehck = firstGridCheck(h5path, acvs_save_path=acvs_save_path)
    if notMatchFirstCehck != 0:
        secondGridCheck(h5path=h5path, acvs_save_path=acvs_save_path)
    print('\n')


if __name__ == "__main__":
    k_grid_summary(h5path='../', acvs_save_path='../save_hBN_symm/acvs.save')

