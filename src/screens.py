from typing import Optional

import numpy as np

class Screen:
    def __init__(
        self,
        width: int,
        height: int,
        pitch: float,
        yaw: float,
        shift: np.ndarray,
        radius: Optional[float]=None
    ) -> None:
        """
        Initialize the Screen with relevant properties. Also rotation matrix R
        such that i_local = R @ i + shift$, where i is the x unit vector in the
        global coordinate system, and i_local is the x unit vector in the local
        coordinate system of the screen. Note that R^{-1} = R.T.

        Parameters
        ----------
        width
            Width of the screen, in pixels.
        height
            Height of the screen, in pixels.
        pitch
            Angle of rotation of the screen about the x (along-width) axis, in
            degrees. A laptop screen at a right angle to its keyboard has zero
            pitch, and closing the laptop (bringing the top towards you) is
            positive pitch.
        yaw
            Angle of rotation of the screen about the y (along-height) axis, in
            degrees. A screen facing the viewer dead-on has a yaw of zero, and
            rotations clockwise from above (e.g., showing the screen to a friend
            on your left) is positive yaw.
        shift
            Center of the screen, in global coordinates.
        radius
            Radius of curvature of the screen. If `None`, the screen is flat.

        """

        self.width = width
        self.height = height
        self.shift = shift
        self.radius = radius

        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)

        pitch_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), np.sin(pitch)],
            [0, -np.sin(pitch), np.cos(pitch)]
        ])

        yaw_matrix = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])

        self.rotation = yaw_matrix @ pitch_matrix

    def to_global(self, pixels: np.ndarray) -> np.ndarray:
        """
        Convert the pixel coordinates of a point on the screen to the global
        Cartesian coordinate system.

        Parameters
        ----------
        pixels
            Array whose first and second elements correspond to the number of
            pixels to the right of and above the bottom left corner of the
            screen, respectively.
        
        Returns
        -------
        np.ndarray
            Array whose three elements are the coordinates of the point on the
            screen expressed in the global coordinate system.

        """

        x_shift = pixels[0] - self.width / 2
        y_shift = pixels[1] - self.height / 2
        
        if self.radius is None:
            x_local = x_shift
            z_local = 0

        else:
            angle = x_shift / self.radius
            x_local = self.radius * np.sin(angle)
            z_local = self.radius * (1 - np.cos(angle))

        local = np.array([x_local, y_shift, z_local])
        return self.rotation.T @ local + self.shift
