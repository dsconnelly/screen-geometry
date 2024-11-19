import numpy as np

from src.screens import Screen

def test_flat_screen_in_front() -> None:
    screen = Screen(
        width=2, height=1,
        pitch=0, yaw=0,
        shift=np.array([0, 0, -1]),
    )

    out = screen.to_global(np.array([1, 0.5]))
    assert np.allclose(out, screen.shift)

    out = screen.to_global(np.array([0, 0]))
    assert np.allclose(out, np.array([-1, -0.5, -1]))

def test_flat_screen_to_left() -> None:
    screen = Screen(
        width=2, height=1,
        pitch=0, yaw=-90,
        shift=np.array([-1, 0, 0])
    )

    out = screen.to_global(np.array([1, 0.5]))
    assert np.allclose(out, screen.shift)

    out = screen.to_global(np.array([0, 0]))
    assert np.allclose(out, np.array([-1, -0.5, 1]))

def test_flat_screen_to_front_right() -> None:
    screen = Screen(
        width=2, height=1,
        pitch=0, yaw=45,
        shift=np.array([1, 0, -1])
    )

    out = screen.to_global([1, 0.5])
    assert np.allclose(out, screen.shift)

    a = 1 / np.sqrt(2)
    out = screen.to_global([0, 0])
    assert np.allclose(out, np.array([1 - a, -0.5, -1 - a]))

def test_pitched_flat_screen_in_front() -> None:
    screen = Screen(
        width=2, height=1,
        pitch=45, yaw=0,
        shift=np.array([0, 0, -1])
    )

    out = screen.to_global([1, 0.5])
    assert np.allclose(out, screen.shift)

    a = 1 / (2 * np.sqrt(2))
    out = screen.to_global([0, 0])
    assert np.allclose(out, np.array([-1, -a, -1 - a]))
