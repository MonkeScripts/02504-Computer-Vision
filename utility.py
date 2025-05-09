import numpy as np


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


def projectpoints(K: np.array, R: np.array, t: np.array, Q: np.array):
    """
    Obtains projected 2D coordinates from world coordinates

    Args:
        K (np.array) intrinsics matrix
        R (np.array) extrinsic rotation matrix
        t (np.array) extrinsic translation matrix
        Q (np.array) homogeneous input points in world coordinates

    Return:
        np.array projected 2D points
    """
    # Projection matrix = K[R t] Q
    extrinsics = np.concatenate((R, t), axis=1)
    return K @ extrinsics @ Q


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
