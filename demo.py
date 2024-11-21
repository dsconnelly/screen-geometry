import matplotlib.pyplot as plt
import numpy as np

from src.scenes import find_screen_and_point
from src.screens import Screen
from src.visualization import add_screen, init_plot

if __name__ == '__main__':
    screens = [
        Screen(
            width=3, height=2,
            pitch=0, yaw=0,
            shift=np.array([0, 0, -2.5])
        ),

        Screen(
            width=1, height=3,
            pitch=0, yaw=-50,
            shift=np.array([-2, 0, -1])
        ),

        Screen(
            width=3, height=1,
            pitch=10, yaw=45,
            shift=np.array([2, 0.5, -1.5]),
            radius=4
        )
    ]

    ax = init_plot()
    colors = ['royalblue', 'forestgreen', 'tab:red']

    for screen, color in zip(screens, colors):
        add_screen(ax, screen, color)

    fix_pixels = np.array([0.5, 0.5])
    fix_screen = screens[1]

    fix_point = fix_screen.to_global(fix_pixels)
    x, y, z = fix_point
    ax.quiver(0, 0, 0, z, x, y, color='k', label='fixation')

    angles = np.array([-85, 40])
    screen, pixels = find_screen_and_point(
        fix_pixels,
        fix_screen,
        screens,
        angles
    )

    x, y, z = screen.to_global(pixels)
    ax.quiver(
        0, 0, 0, z, x, y, 
        color='gold', 
        label='($-85^\\circ$, $40^\\circ$)'
    )
    print(screen is screens[-1])


    angles = np.array([-55, 10])
    screen, pixels = find_screen_and_point(
        fix_pixels,
        fix_screen,
        screens,
        angles
    )

    x, y, z = screen.to_global(pixels)
    ax.quiver(
        0, 0, 0, z, x, y, 
        color='darkviolet', 
        label='($-55^\\circ$, $10^\\circ$)'
    )
    print(screen is screens[0])

    ax.legend()
    plt.show()