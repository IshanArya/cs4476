import numpy as np


def compute_homography(t1, t2):
    """
    Computes the Homography matrix for corresponding image points t1 and t2

    The function should take a list of N ≥ 4 pairs of corresponding points 
    from the two views, where each point is specified with its 2d image 
    coordinates.

    Inputs:
    - t1: Nx2 matrix of image points from view 1
    - t2: Nx2 matrix of image points from view 2

    Returns a tuple of:
    - H: 3x3 Homography matrix
    """
    H = np.eye(3)
    #############################################################################
    # TODO: Compute the Homography matrix H for corresponding image points t1, t2
    #############################################################################

    N = t1.shape[0]

    p = np.append(t1, np.ones((N, 1)), 1)
    q = np.append(t2, np.ones((N, 1)), 1)

    L = np.zeros((N * 2, 9))

    for i in range(N):
      L[i * 2, 0:3] = p[i]
      L[i * 2, 3:6] = np.zeros(3)
      L[i * 2, 6:9] = -1 * q[i, 0] * p[i]

      L[i * 2 + 1, 0:3] = np.zeros(3)
      L[i * 2 + 1, 3:6] = p[i]
      L[i * 2 + 1, 6:9] = -1 * q[i, 1] * p[i]

    M = np.matmul(np.transpose(L), L)

    _, _, v = np.linalg.svd(M)

    H = v[-1, :].reshape((3, 3))

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return H
