import numpy as np
import pytest

from src.screens import NoIntersection, Screen

def test_flat_screen_intersections() -> None:
    screen = Screen(
        width=5, height=1,
        pitch=0, yaw=0,
        shift=np.array([0, 0, -3])
    )

    point = screen.find_intersect(np.array([0, 0, -1]))
    assert np.allclose(point, np.array([2.5, 0.5]))

    a = 1 / np.sqrt(3)
    point = screen.find_intersect(np.array([a, 0, -1]))
    assert np.allclose(point, np.array([2.5 + 3 * a, 0.5]))

    a = 1 / np.sqrt(2)
    with pytest.raises(NoIntersection):
        point = screen.find_intersect(np.array([0, a, -a]))
