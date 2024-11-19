import numpy as np

from src.screens import Screen

def test_curved_screen_in_front() -> None:
    screen = Screen(
        width=2, height=1,
        pitch=0, yaw=0,
        shift=np.array([0, 0, -2]),
        radius=1
    )

    out = screen.to_global(np.array([1, 0.5]))
    assert np.allclose(out, screen.shift)

    out = screen.to_global(np.array([0, 0]))
    assert np.allclose(out, np.array([-np.sin(1), -0.5, -1 - np.cos(1)]))

def test_very_curved_screen_in_front() -> None:
    screen = Screen(
        width=2, height=1,
        pitch=0, yaw=0,
        shift=np.array([0, 0, -1]),
        radius=2
    )

    out = screen.to_global(np.array([1, 0.5]))
    assert np.allclose(out, screen.shift)

    out = screen.to_global(np.array([0, 0]))
    target = np.array([-2 * np.sin(0.5), -0.5, -1 + 2 * (1 - np.cos(0.5))])
    assert np.allclose(out, target)

def test_pitched_curved_screen_above_in_front() -> None:
    shift = np.array([0, 2, -2])
    r = np.sqrt((shift ** 2).sum())
    C = 2 * np.pi * r

    screen = Screen(
        width=(C / 2), height=1,
        pitch=45, yaw=0,
        shift=shift,
        radius=r
    )

    out = screen.to_global(np.array([C / 4, 0.5]))
    assert np.allclose(out, screen.shift)

    out = screen.to_global(np.array([0, 0.5]))
    assert np.allclose(out, np.array([-r, 0, 0]))
    