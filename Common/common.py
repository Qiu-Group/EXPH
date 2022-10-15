import numpy as np
from Common.progress import ProgressBar

#############################################################################################################
# This includes some frequently used but fundamental functions:                                             #
# (1) equivalence_even_order(A,B,tolerance=1E-4)                                                            #
# (2) equivalence_no_order(A,B,tolerance=1E-4)                                                              #
# (3) move_k_back_to_BZ_1(A)                                                                                #
# (4) isDoubleCountK(A)                                                                                     #
# (5) frac2carte(a0, fractional_point)                                                                      #
#############################################################################################################

def equivalence_order(A, B, tolerance=1E-4):
    """
    even_order: means the order of A and B should be same
    :param A/B: any numpy array A and B
    :param tolerance: tolerance of difference
    :return: True or False
    """
    # firstly compare dimension
    if A.shape != B.shape:
        return False
    # if True: each element
    delta_1 = np.where(np.abs((A - B)) >= tolerance, 1, 0)
    if 1 in delta_1:
        return False
    else:
        return True


def equivalence_no_order(A, B, tolerance=1E-4, mute=True):
    """
    no_order: means the order of A and B could be different, but they should have one by one equivalence
    :param A/B: any numpy array A and B
    :param tolerance: tolerance of difference
    :return: True or False
    """
    # firstly compare dimension
    # print('start check two sets of k-grid')
    if np.sum(A.shape)==3:# if this is a vector
        equivalence_order(A,B,tolerance)

    if A.shape != B.shape:
        if not mute:
            print('ths dimension of two grid not matching')
        return False
    pop_index = []
    # progress = ProgressBar(A.shape[0], fmt=ProgressBar.FULL)
    for i in range(A.shape[0]):
        # progress.current += 1
        # progress()
        count = 0 # this is a flag to determine if we find a point matching with A[i]
        for j in range(B.shape[0]):
            if equivalence_order(A[i], B[j], tolerance):
                if j in pop_index:
                    if not mute:
                        print('we find multiple map from A to B')
                    return False
                else:
                    pop_index.append(j)
                    count = 1 # we find a unique point in B set matching with point in A
                    continue
        if count == 0: # we don't find any point in B matching with A point
            if not mute:
                print('we don\'t find any point in B matching with:', A[i])
            return False
    if not mute:
        print("two kgrid set match")
    return True

def move_k_back_to_BZ_1(A):
    """
    It doesnot change the order of A
    :param A: A could be any numpy array, but IT HAS TO BE FRACTIONAL coordinate
    :return: A_BZ: 0 <= A_BZ[()] < 1
    """
    temp_0 = np.where(np.abs(A - 1) < 0.00001, 1.0, A)
    temp_0_ = np.where(np.abs(temp_0 - 0.0) < 0.00001, 0.0, temp_0)
    temp_1 = np.where(temp_0_ >= 0.999999999, temp_0_ - 1, temp_0_)
    res = np.where(temp_1 < -0.000000001, temp_1 + 1, temp_1)
    # temp_1 = np.where(A >= 0.99999999999999999999999999999, A - 1, A)
    # res = np.where(temp_1 < -0.00000000000000000000000000001, temp_1 + 1, temp_1)

    if isDoubleCountK(res):
        raise Exception("There is a double count in A")
    return res

def isDoubleCountK(A):
    """
    :param A: a k-grid list: A.shape = (n,3)
    :return: True or False
    """
    if np.sum(A.shape)==3:# if this is a vector
        return False

    dbc_tolerance = 5E-6
    for i in range(A.shape[0] - 1):
        for j in range(i + 1, A.shape[0]):
            if np.linalg.norm(A[i] - A[j]) < dbc_tolerance:
                return True
    return False

def frac2carte(a0, fractional_point):
    """
    :param a0: a0.shape = (3,3), the reciprocal lattice (Cartesian)
    :param fractional_point: fractional_point.shape = (n,3) (coordinate)
    :return:
    """
    if a0.shape[0] != 3 or a0.shape[1] != 3:
        raise Exception("a0.shape[0] != 3 or a0.shape[1] != 3")
    if np.sum(fractional_point.shape) == 3:
        lenth = 1
    else:
        lenth = fractional_point.shape[0]
    return (a0.T@fractional_point.T).T