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

def test_normals() -> None:
    screen = Screen(
        width=2, height=1,
        pitch=0, yaw=0,
        shift=np.array([0, 0, -2])
    )

    assert np.allclose(screen.normal, np.array([0, 0, 1]))

    screen = Screen(
        width=3, height=1,
        pitch=45, yaw=0,
        shift=np.array([0, 0, -2])
    )

    a = 1 / np.sqrt(2)
    assert np.allclose(screen.normal, np.array([0, -a, a]))

    screen = Screen(
        width=3, height=1,
        pitch=0, yaw=-45,
        shift=np.array([0, 0, -2])
    )

    assert np.allclose(screen.normal, np.array([a, 0, a]))

    screen = Screen(
        width=3, height=1,
        pitch=27, yaw=-13,
        shift = np.array([2, 1.1, -4])
    )

    u = screen.to_global(np.array([0.3, 0.3]))
    v = screen.to_global(np.array([0.3, 0.31]))
    w = screen.to_global(np.array([0.31, 0.3]))

    assert np.allclose((v - u) @ screen.normal, 0)
    assert np.allclose((w - u) @ screen.normal, 0)