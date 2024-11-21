import numpy as np

def get_angle(p: np.ndarray, q: np.ndarray) -> float:
    """
    Get the unsigned angle between two vectors in global coordinates.

    Parameters
    ----------
    p, q
        Two vectors, with three elements each.

    Returns
    -------
    float
        Angle between `p` and `q`.

    """

    cosine = (p @ q) / np.linalg.norm(p) / np.linalg.norm(q)
    return np.arccos(cosine)

def get_spherical(point: np.ndarray) -> np.ndarray:
    """
    Get the spherical angles of a point in 3D space.

    Parameters
    ----------
    point
        Array of 3D coordinates of the point in question.

    Returns
    -------
    np.ndarray
        Spherical angles (theta, phi).

    """

    x, y, z = point
    theta = np.arctan2(x, z)
    phi = np.arctan2(y, np.sqrt(x ** 2 + z ** 2))

    return np.array([theta, phi])
