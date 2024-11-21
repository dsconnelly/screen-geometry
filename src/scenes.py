import numpy as np

from .screens import NoIntersection, Screen
from .utils import get_spherical

def find_screen_and_point(
    fix_pixels: np.ndarray,
    fix_screen: Screen,
    screens: list[Screen],
    angles: np.ndarray
) -> tuple[Screen, np.ndarray]:
    """
    Find the appropriate screen and pixels on that screen, given the fixation
    and the desired viewing angles.

    Parameters
    ----------
    fix_pixels
        Pixel coordinates of the fixation.
    fix_screen
        Screen on which the fixation is being shown.
    screens
        Available screens.
    angles
        Desired viewing angles (theta, phi).

    Returns
    -------
    Screen
        Screen on which to show the other point.
    np.ndarray
        Pixel coordinates of the new point on that screen.

    Raises
    ------
    RuntimeError
        If no screen contains a point with the desired viewing angles.

    """

    fix_point = fix_screen.to_global(fix_pixels)
    theta, phi = np.deg2rad(angles) + get_spherical(fix_point)

    ray = np.array([
        np.cos(phi) * np.sin(theta),
        np.sin(phi),
        np.cos(phi) * np.cos(theta)
    ])

    best_screen = None
    best_pixels = None
    min_dist = np.inf

    for screen in screens:
        try:
            point, pixels = screen.find_intersect(ray)
            dist = np.linalg.norm(point)

            if dist < min_dist:
                best_screen = screen
                best_pixels = pixels
                min_dist = dist

        except NoIntersection:
            continue

    if best_screen is None:
        raise RuntimeError('No intersection on any screen was found')

    return best_screen, best_pixels
