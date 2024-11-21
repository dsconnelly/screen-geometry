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

    # Get the location of the fixation in 3D global coordinates.
    fix_point = fix_screen.to_global(fix_pixels)

    # Get the angular direction of the target from the eye by getting the angles
    # of the fixation and adding on the desired viewing angles.
    theta, phi = np.deg2rad(angles) + get_spherical(fix_point)

    # Convert back out of spherical coordinates to get a ray pointing in the
    # direction in which the target should appear.
    ray = np.array([
        np.cos(phi) * np.sin(theta),
        np.sin(phi),
        np.cos(phi) * np.cos(theta)
    ])

    best_screen = None
    best_pixels = None
    min_dist = np.inf

    # Iterate over all available screens. Since in this code the origin of the
    # global system is just the eye, we can get the distance of each possible
    # intersection point as just the norm of the vector. We save the closest.
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
        # None of the screens found an intersection.
        raise RuntimeError('No intersection on any screen was found')

    return best_screen, best_pixels
