from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes

from .screens import Screen

def init_plot() -> Axes:
    """
    Initialize a plot to show 3D screen geometry.

    Returns
    -------
    Axes
        Three-dimensional axis, ready for plotting.

    """

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter([0], [0], [0], color='k', s=20)

    ax.set_xlim(-3.5, 0.5)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.set_zlabel('y')

    return ax

def add_screen(ax: Axes, screen: Screen, color: str='k') -> None:
    """
    Visualize a screen on an initialized axis.

    Parameters
    ----------
    ax
        Three-dimensional axis as returned by `init_plot`.
    screen
        Screen object to add to the scene.
    color
        Color to render the screen in.

    """

    xs = np.linspace(0, screen.width, 100)
    ys = np.linspace(0, screen.height, 100)
    points = [screen.to_global(np.array([x, y])) for x, y in product(xs, ys)]
    
    x, y, z = np.vstack(points).T
    ax.scatter(z, x, y, color=color, s=2)
