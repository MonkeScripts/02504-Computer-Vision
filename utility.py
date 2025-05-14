import numpy as np
import itertools as it


def box3d(n=16):
    """Generate 3D points inside a cube with n-points along each edge"""
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i,) * n, (j,) * n, N])))
    return np.hstack(points) / 2


def point_line_distance(line: np.ndarray, p: np.ndarray):
    """
    Calculate shortest distance d between line l and 2D homogenous point p.

    Args:
        line: homogenous line, shape (3, 1)
        p: 3x1 vector, shape (3, 1)

    Returns:
        d (float): distance
    """
    if p.shape != (3, 1):
        raise ValueError(f"p must be a 3x1 homogenous vector, current shape {p.shape}")
    if line.shape != (3, 1):
        raise ValueError(f"line must be a 3x1 homogenous vector, current shap {line.shape}")

    d = abs(line.T @ p) / (abs(p[2]) * np.sqrt(line[0] ** 2 + line[1] ** 2))
    return d


def Pi(x: np.array):
    """
    Converts homogeneous to inhomogeneous coordinates
    Args:
        x (np.array) homogeneous coordinate

    Return:
        np.array converted inhomogeneous coordinate
    """

    return x[:-1] / x[-1]


def Piinv(x: np.array):
    """
    Converts inhomogeneous to homogeneous coordinates

    Args:
        x (np.array) inhomogeneous coordinate

    Return:
        np.array converted homogeneous coordinate
    """
    if x.ndim == 1:
        return np.concatenate((x, np.ones(1)))
    return np.vstack((x, np.ones((1, x.shape[1]))))


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray):
    """
    Create a projection matrix from camera parameters.

    Args:
        K (np.array): intrinsic camera matrix, shape (3, 3)
        R (np.array): rotation matrix, shape (3, 3)
        t (np.array): translation matrix, shape (3, 1)

    Returns:
        P (np.array): projection matrix, shape (3, 4)
    """
    if K.shape != (3, 3):
        raise ValueError("K must be a 3x3 matrix")
    if R.shape != (3, 3):
        raise ValueError("R must be a 3x3 matrix")
    if t.shape != (3, 1):
        raise ValueError("t must be a 3x1 matrix")

    P = K @ np.hstack((R, t))
    return P


def dist(point: np.array, distCoeffs: np.array, poly_func):
    """
    Returns a distorted version of x based on radial distortion
    With polynomial coefficients in distCoeffs

    Args:
        point (np.array): Input inhomogeneous 2d points
        distCoeffs (np.array): radial distortion polynomial coefficients
        poly_func (function): Function that computes polynomial

    Return:
        np.array of distorted image in inhomogeneous coordinates
    """
    point_x = point[0]
    point_y = point[1]
    squared = np.sqrt(point_x**2 + point_y**2)
    print(f"Undergoing distortion with coeffs: {distCoeffs}")
    distortion_factor = poly_func(squared, distCoeffs)
    distorted_point = point * distortion_factor
    return distorted_point


def projectpoints(K: np.array, R: np.array, t: np.array, distCoeffs: np.array, Q: np.array):
    """
    Obtains projected 2D coordinates from world coordinates

    Args:
        K (np.array): intrinsics matrix
        R (np.array): extrinsic rotation matrix
        t (np.array): extrinsic translation matrix
        distCoeffs: (np.array) distortion coefficient matrix
        Q (np.array): homogeneous input points in world coordinates

    Return:
        np.array projected 2D points with distortion
    """
    # Projection matrix = K[R t] Q
    extrinsics = np.concatenate((R, t), axis=1)
    poly_lambda = (
        lambda squared, coeffs: 1 + coeffs[0] * (squared ** 2)
    )
    distorted_point = dist(Pi(extrinsics @ Q), distCoeffs, poly_lambda)
    print(f"distorted_point is {distorted_point}")
    return K @ Piinv(distorted_point)


def skew(x: np.array):
    """
    This function returns a numpy array with the skew symmetric cross product matrix for vector.
    the skew symmetric cross product matrix is defined such that
    np.cross(a, b) = np.dot(skew(a), b)
    https://stackoverflow.com/questions/36915774/form-numpy-array-from-possible-numpy-array

    Args:
        x (np.array): 1x3 matrix

    Return:
        s (np.array): 3x3 skew symmetrix matrix for cross product
    """
    vector = x.ravel()
    s = np.asarray(
        [
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0],
        ]
    )
    return s


def normalize2d(q: np.ndarray):
    """
    Normalize 2D points to have mean 0 and sd 1.

    Args:
        q : 2 x n, 2D points

    Returns
        qn : 2 x n, normalized 2D points
        T : 3 x 3, normalization matrix
    """
    if q.shape[0] != 2:
        raise ValueError("q must have 2 rows")
    if q.shape[1] < 2:
        raise ValueError("At least 2 points are required to normalize")

    mu = np.mean(q, axis=1).reshape(-1, 1)
    mu_x = mu[0].item()
    mu_y = mu[1].item()
    std = np.std(q, axis=1).reshape(-1, 1)
    std_x = std[0].item()
    std_y = std[1].item()
    Tinv = np.array([[std_x, 0, mu_x], [0, std_y, mu_y], [0, 0, 1]])
    T = np.linalg.inv(Tinv)
    qn = T @ Piinv(q)
    qn = Pi(qn)
    return qn, T


def hest(q1: np.ndarray, q2: np.ndarray, normalize=False):
    """
    Calculate the homography matrix from n sets of 2D points
    q1 = H @ q2

    Args:
        q1 : 2 x n, 2D points in the first image
        q2 : 2 x n, 2D points in the second image
        normalize : bool, whether to normalize the points

    Returns:
        H : 3 x 3, homography matrix
    """
    if q1.shape[1] != q2.shape[1]:
        raise ValueError("Number of points in q1 and q2 must be equal")
    if q1.shape[1] < 4:
        raise ValueError(
            "At least 4 points are required to estimate a homography",
        )
    if q1.shape[0] != 2 or q2.shape[0] != 2:
        raise ValueError("q1 and q2 must have 2 rows")

    if normalize:
        q1, T1 = normalize2d(q1)
        q2, T2 = normalize2d(q2)

    n = q1.shape[1]
    B = []
    for i in range(n):
        x1, y1 = q1[:, i]
        x2, y2 = q2[:, i]
        Bi = np.array(
            [
                [0, -x2, x2 * y1, 0, -y2, y2 * y1, 0, -1, y1],
                [x2, 0, -x2 * x1, y2, 0, -y2 * x1, 1, 0, -x1],
                [-x2 * y1, x2 * x1, 0, -y2 * y1, y2 * x1, 0, -y1, x1, 0],
            ],
        )
        B.append(Bi)
    B = np.array(B).reshape(-1, 9)
    U, S, Vt = np.linalg.svd(B)
    H = Vt[-1].reshape(3, 3)
    H = H.T
    if normalize:
        H = np.linalg.inv(T1) @ H @ T2
    return H


def triangulate(pixel_coords: np.array, proj_matrices: np.array):
    """
    Given a list of pixel coordinates and projection matrices, triangulate to a common 3D point
    Args:
        pixel_coords (np.array): list of pixel coordinates
        proj_matrices (np.array): list of projection matrices

    Return:
        triangle (np.array): triangulated 3D point
    """
    n = pixel_coords.shape[0]
    # B_stack = []
    B_stack = np.zeros((n * 2, 4))
    for i in range(n):
        x, y = pixel_coords[i]
        proj_matrix = proj_matrices[i]
        B = np.asarray(
            [
                proj_matrix[2, :] * x - proj_matrix[0, :],
                proj_matrix[2, :] * y - proj_matrix[1, :],
            ]
        )
        B_stack = np.vstack((B_stack, B))
    U, S, Vt = np.linalg.svd(B_stack)
    # Get the smallest vector
    print(f"Vt is {Vt}")
    triangle = Vt[-1, :]
    triangle /= triangle[-1]
    return triangle


# Ex 3.3
def essential_matrix(R: np.ndarray, t: np.ndarray):
    """
    Returns the essential matrix.

    Args:
        R : 3x3 matrix, rotation matrix
        t : 3x1 matrix, translation matrix

    Returns:
        E : 3x3 matrix, essential matrix
    """
    return skew(t) @ R


def fundamental_matrix(
    K1: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    K2: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
):
    """
    Returns the fundamental matrix, assuming camera 1 coordinates are
    on top of global coordinates.

    Args:
        K1 : 3x3 matrix, intrinsic matrix of camera 1
        R1 : 3x3 matrix, rotation matrix of camera 1
        t1 : 3x1 matrix, translation matrix of camera 1
        K2 : 3x3 matrix, intrinsic matrix of camera 2
        R2 : 3x3 matrix, rotation matrix of camera 2
        t2 : 3x1 matrix, translation matrix of camera 2

    Returns:
        F : 3x3 matrix, fundamental matrix
    """
    if R1.shape != (3, 3) or R2.shape != (3, 3):
        raise ValueError("R1 and R2 must be 3x3 matrices")
    if t1.shape == (3,) or t2.shape == (3,):
        t1 = t1.reshape(-1, 1)
        t2 = t2.reshape(-1, 1)
    if t1.shape != (3, 1) or t2.shape != (3, 1):
        raise ValueError("t1 and t2 must be 3x1 matrices")
    if K1.shape != (3, 3) or K2.shape != (3, 3):
        raise ValueError("K1 and K2 must be 3x3 matrices")

    # When the {camera1} and {camera2} are not aligned with {global}
    R_tilde = R2 @ R1.T
    t_tilde = t2 - R_tilde @ t1

    E = essential_matrix(R_tilde, t_tilde)
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F

import scipy.optimize


def triangulate_nonlin(pixel_coords: np.array, proj_matrices: np.array):
    """
    Given a list of pixel coordinates and projection matrices, triangulate to a common 3D point using a non linear approach
    Args:
        pixel_coords (np.array): list of pixel coordinates in inhomogeneous coordinates
        proj_matrices (np.array): list of projection matrices

    Return:
        triangle (np.array): triangulated 3D point
    """

    def compute_residuals(Q: np.array):
        """
        Compute residuals between projected points and observed pixel coordinates.
        Args:
            Q (np.array): Current estimate of 3D point (x, y, z) in homogeneous coordinates

        Return:
            residuals (np.array): Vector of residuals (differences between projected and observed points)
        """
        if Q.shape[0] == 3:
            Q = Piinv(Q)
        residuals = np.zeros(2 * len(pixel_coords))
        for i, q in enumerate(pixel_coords):
            projected_point = Pi(proj_matrices[i, :, :] @ Q)
            # Flatten
            diff_vector = q.reshape(-1) - projected_point.reshape(-1)
            residuals[2 * i : 2 * (i + 1)] = diff_vector
        return residuals

    # Initial guess
    x0 = triangulate(pixel_coords, proj_matrices).reshape(-1)
    least_error_3D = scipy.optimize.least_squares(compute_residuals, x0)
    return least_error_3D.x

def pest(Q: np.ndarray, q: np.ndarray, normalize=False):
    """
    Estimate projection matrix using direct linear transformation.

    Args:
        Q : 3 x n array of 3D points
        q : 2 x n array of 2D points
        normalize : bool, whether to normalize the 2D points

    Returns:
        P : 3 x 4 projection matrix
    """
    if Q.shape[0] != 3:
        raise ValueError("Q must be a 3 x n array of 3D points")
    if q.shape[0] != 2:
        raise ValueError("q must be a 2 x n array of 2D points")

    if normalize:
        q, T = normalize2d(q)

    q = Piinv(q)  # 3 x n
    Q = Piinv(Q)  # 4 x n
    n = Q.shape[1]  # number of points
    B = []
    for i in range(n):
        Qi = Q[:, i]
        qi = q[:, i]
        Bi = np.kron(Qi, skew(qi))
        B.append(Bi)
    B = np.array(B).reshape(3 * n, 12)
    U, S, Vt = np.linalg.svd(B)
    P = Vt[-1].reshape(4, 3)
    P = P.T
    if normalize:
        P = np.linalg.inv(T) @ P
    return P


def fit_line(p1: np.ndarray, p2: np.ndarray):
    """
    Fits a line given 2 points.

    Args:
        p1, p2 (np.array) : 2x1 inhomogenous coordinates

    Returns:
        l : 3x1, line in homogenous coordinates
    """
    if p1.shape == (2,):
        p1 = p1.reshape(2, 1)
        p2 = p2.reshape(2, 1)
    if p1.shape != (2, 1) or p2.shape != (2, 1):
        raise ValueError("Points must be 2x1 np.array")

    p1h = Piinv(p1)
    p2h = Piinv(p2)
    # cross() requires input as vectors
    l = np.cross(p1h.squeeze(), p2h.squeeze())
    return l


def find_inliers_outliers(l: np.ndarray, points: np.ndarray, tau: float):
    """
    Args:
        l : equation of line in homogenous coordinates
        points : 2xn, set of 2D points
        tau : threshold for inliners

    Returns:
        inliners (np.array) : 2xa, set of inliner points
        outliers (np.array) : 2xb, set of outlier points
    """
    inliners = []
    outliers = []
    for p in points.T:
        p = p.reshape(2, 1)
        ph = Piinv(p)
        d = abs(l.T @ ph) / (abs(ph[2]) * np.sqrt(l[0] ** 2 + l[1] ** 2))
        if d <= tau:  # inliner
            inliners.append(p)
        else:  # outlier
            outliers.append(p)
    inliners = np.array(inliners).squeeze().T
    outliers = np.array(outliers).squeeze().T
    return inliners, outliers
