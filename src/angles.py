import numpy as np

from .screens import Screen

def get_angle(
    screen_1: Screen,
    screen_2: Screen,
    pixels_1: np.ndarray,
    pixels_2: np.ndarray
) -> float:
    """
    
    """

    point_1 = screen_1.to_global(pixels_1)
    point_2 = screen_2.to_global(pixels_2)
    norm_1 = np.linalg.norm(point_1)
    norm_2 = np.linalg.norm(point_2)
    
    cosine = (point_1 @ point_2) / norm_1 / norm_2
    return np.rad2deg(np.arccos(cosine))
