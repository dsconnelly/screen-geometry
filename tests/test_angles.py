import numpy as np

from src.angles import get_angle
from src.screens import Screen

def test_points_on_flat_screen() -> None:
    screen = Screen(
        width=2, height=1,
        pitch=0, yaw=0,
        shift=np.array([0, 0, -3])
    )

    pixels_1 = np.array([0, 0.5])
    pixels_2 = np.array([1, 0.5])
    pixels_3 = np.array([1, 1])

    out = get_angle(screen, screen, pixels_1, pixels_2)
    assert np.allclose(out, np.rad2deg(np.arctan(1 / 3)))

    out = get_angle(screen, screen, pixels_2, pixels_3)
    assert np.allclose(out, np.rad2deg(np.arctan(1 / 6)))

def test_points_on_pitched_flat_screen() -> None:
    screen = Screen(
        width=2, height=1,
        pitch=30, yaw=0,
        shift=np.array([0, 0, -3])
    )

    pixels_1 = np.array([0, 0.5])
    pixels_2 = np.array([1, 0.5])
    pixels_3 = np.array([1, 1])

    out = get_angle(screen, screen, pixels_1, pixels_2)
    assert np.allclose(out, np.rad2deg(np.arctan(1 / 3)))

    out = get_angle(screen, screen, pixels_2, pixels_3)
    target = np.arctan(np.cos(np.pi / 6) / (6 - np.sin(np.pi / 6)))
    assert np.allclose(out, np.rad2deg(target))

def test_points_on_two_flat_screens() -> None:
    screen_1 = Screen(
        width=2, height=1,
        pitch=0, yaw=0,
        shift=np.array([0, 0, -3])
    )

    screen_2 = Screen(
        width=3, height=1,
        pitch=0, yaw=90,
        shift=np.array([2, 0, 0])
    )

    pixels_1 = np.array([1, 0.5])
    pixels_2 = np.array([1.5, 0.5])
    pixels_3 = np.array([0, 0.5])
    pixels_4 = np.array([3, 0.5])

    out = get_angle(screen_1, screen_2, pixels_1, pixels_2)
    assert np.allclose(out, 90)

    out = get_angle(screen_1, screen_2, pixels_1, pixels_3)
    assert np.allclose(out, np.rad2deg(np.arctan(4 / 3)))

    out = get_angle(screen_1, screen_2, pixels_1, pixels_4)
    assert np.allclose(out, 90 + np.rad2deg(np.arctan(3 / 4)))