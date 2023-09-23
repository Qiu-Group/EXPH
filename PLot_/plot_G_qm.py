import numpy as np
import h5py as h5
import os

def get_fine_q():
    f = h5.File('G_qm.h5', 'r')
    kpt = f['q_fi_frac'][()]
    f.close()
    return  kpt

def get_highsymmetry_index_kmap(high_symm="0.0 0.0 0.0 , 0.5 0.0 0.0"):
    """
    :return: [index1, index2, ...] which is index for high symmetry k-points and lines
    """
    print('------------finding high symmetry path------------')
    tolerence = 1E-6
    assert os.path.isfile('G_qm.h5')
    kmap_points = get_fine_q()[:,:2]
    nq = kmap_points.shape[0]

    # (i) get the index of high symmetry point you set
    high_symm_list = high_symm.split(',')
    high_symm_kpoints_kmap_index = []  # [high_symm1, high_symm2, ...]
    for h_kpt in high_symm_list:
        found_high = False
        h_kpt_array = np.fromstring(h_kpt.strip(), dtype=float, sep=' ')[:2]

        True_False_matrix_to_be_determined = (np.abs(kmap_points - h_kpt_array) < tolerence)
        for i in range(kmap_points.shape[0]):
            if np.all(True_False_matrix_to_be_determined[i]):
                high_symm_kpoints_kmap_index.append([i, i // int(np.sqrt(nq)), i % int(np.sqrt(nq))])
                print('index of point: %.5f %.5f 0.00000 ' % (h_kpt_array[0], h_kpt_array[1]), ' in kmap :', i,
                      ' [%s,%s]' % (i // int(np.sqrt(nq)), i % int(np.sqrt(nq))))
                found_high = True
                break
        if not found_high:
            raise Exception('kpoint: ' + h_kpt + ' can not be found in kkqQmap.dat')
    # print(high_symm_kpoints_kmap_index)

    # (ii) get the index list of high symmetry line

    res_matrix_index = []
    res_kmap_index = []
    for i in range(len(high_symm_kpoints_kmap_index) - 1):
        # print('\n-------')
        temp_start_point = np.array(high_symm_kpoints_kmap_index[i][1:])
        temp_end_point = np.array(high_symm_kpoints_kmap_index[i + 1][1:])

        x_range = np.linspace(temp_start_point[0],
                              temp_end_point[0] + 1 if temp_start_point[0] > temp_end_point[0] else temp_end_point[
                                                                                                        0] - 1,
                              np.abs(temp_end_point[0] - temp_start_point[0]))
        y_range = np.linspace(temp_start_point[1],
                              temp_end_point[1] + 1 if temp_start_point[1] > temp_end_point[1] else temp_end_point[
                                                                                                        1] - 1,
                              np.abs(temp_end_point[1] - temp_start_point[1]))

        # print(x_range, y_range)

        if len(x_range) == 0:
            x_range = np.ones_like(y_range) * temp_start_point[0]
        if len(y_range) == 0:
            y_range = np.ones_like(x_range) * temp_start_point[1]

        if max(len(y_range), len(x_range)) % min(len(x_range), len(y_range)) == 0:
            single_period = max(len(y_range), len(x_range)) // min(len(x_range), len(y_range))
            if len(y_range) > len(x_range):
                y_range = np.linspace(temp_start_point[1],
                                      temp_end_point[1] + single_period if temp_start_point[1] > temp_end_point[1] else
                                      temp_end_point[1] - single_period, len(x_range))
            elif len(x_range) > len(y_range):
                x_range = np.linspace(temp_start_point[0],
                                      temp_end_point[0] + single_period if temp_start_point[0] > temp_end_point[0] else
                                      temp_end_point[0] - single_period, len(y_range))

            # print(x_range,y_range)
            for j in range(len(x_range)):  # todo: add index
                res_matrix_index.append([int(x_range[j]), int(y_range[j])])
                res_kmap_index.append(int((x_range[j]) * np.sqrt(nq) + y_range[j]))
                # print([int(x_range[j]), int(y_range[j])])
        else:
            res_matrix_index.append([int(x_range[0]), int(y_range[0])])
            res_kmap_index.append(int((x_range[0]) * np.sqrt(nq) + y_range[0]))
            # print([int(x_range[0]), int(y_range[0])])
    # add the last point
    # res_matrix_index.append(high_symm_kpoints_kmap_index[-1][1:])
    # res_kmap_index.append(int(high_symm_kpoints_kmap_index[-1][1]*np.sqrt(nq) + high_symm_kpoints_kmap_index[-1][2]))
    print('high symmetry path includes %s k points' % len(res_kmap_index))
    # print("matrix path:",res_matrix_index)
    print("kmap path", res_kmap_index)
    print('------------finding high symmetry path done------------')
    return res_kmap_index

def plot_g_qm(high_symm, m_start, m_end, mu_start, mu_end):
    f = h5.File("G_qm.h5",'r')
    kmap_path_index = get_highsymmetry_index_kmap(high_symm=high_symm)
    omega = f['Omega_Qm'][kmap_path_index,m_start:m_end]
    gamma_qvm = f['gamma_qvm'][kmap_path_index, mu_start:mu_end ,m_start:m_end].sum(axis=1)
    G_qvm = f['G_qvm'][kmap_path_index, mu_start:mu_end, m_start:m_end].sum(axis=1)

    nk = len(kmap_path_index)
    nS = m_end-m_start

    res = np.zeros((nk*nS, 4))
    res[:, 0] = np.kron(np.ones(nS), np.linspace(0,1,nk))
    res[:, 1] = (omega.transpose()).flatten()
    res[:, 2] = (G_qvm.transpose()).flatten()
    res[:, 3] = (gamma_qvm.transpose()).flatten()

    np.savetxt('G_qm.dat',res)

if __name__ == "__main__":
    # This script could choose high symmetry point and find the high symmetry path
    # Then plot the exciton energy, exciton-phonon, exciton scattering rate along high-symmetry path
    # the required input file is "G_qm.h5"
    high_symm = "0.0 0.0 0.0 , 0.0 0.5 0.0"
    m_start = 0
    m_end = 4 # number of m is m_end - m_start
    mu_start = 0
    mu_end = 9

    plot_g_qm(high_symm=high_symm,m_start=m_start,m_end=m_end,mu_start=mu_start,mu_end=mu_end)
    pass