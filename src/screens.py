from typing import Optional

import numpy as np

class NoIntersection(Exception):
    pass

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
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])

        yaw_matrix = np.array([
            [np.cos(yaw), 0, -np.sin(yaw)],
            [0, 1, 0],
            [np.sin(yaw), 0, np.cos(yaw)]
        ])

        self.rotation = yaw_matrix @ pitch_matrix

        if self.radius is None:
            self.normal = self.rotation[:, 2]

        else:
            p = self.to_global(np.array([self.width / 2, 0]))
            self.base = p + self.radius * self.rotation[:, 2]
            self.axis = self.rotation[:, 1]

            q = self.to_global(np.array([self.width, 0]))
            a, b = p - self.base, q - self.base

            cosine = (a @ b) / np.linalg.norm(a) / np.linalg.norm(b)
            self.theta_max = np.arccos(cosine)

    def find_intersect(self, ray: np.ndarray) -> np.ndarray:
        """
        Find the pixel coordinates of the intersection of a given ray with this
        screen, if such an intersection exists.

        Parameters
        ----------
        ray
            Array whose three elements correspond to the components of a vector
            at the origin pointing in the desired direction.

        Returns
        -------
        np.ndarray
            Array of pixel coordinates at the point of intersection.

        """

        if self.radius is None:
            t = (self.shift @ self.normal) / (ray @ self.normal)

            if t < 0:
                raise NoIntersection

            x = self.rotation[:, 0] @ (t * ray - self.shift) + self.width / 2
            y = self.rotation[:, 1] @ (t * ray - self.shift) + self.height / 2

            if not ((0 <= x <= self.width) and (0 <= y <= self.height)):
                raise NoIntersection
            
            return np.array([x, y])
        
        rxa = np.cross(ray, self.axis)
        t = (rxa @ np.cross(self.base, self.axis) + np.sqrt(
            self.radius ** 2 * (rxa @ rxa) - (self.base @ rxa) ** 2
        )) / (rxa @ rxa)
        
        h = self.axis @ (t * ray - self.base)
        if not (0 <= h <= self.height):
            raise NoIntersection
        
        center = self.base + h * self.rotation[:, 1]
        a = t * ray - center
        b = self.to_global(np.array([self.width / 2, h])) - center

        cosine = (a @ b) / np.linalg.norm(a) / np.linalg.norm(b)
        theta = np.arccos(cosine)

        if theta > self.theta_max:
            raise NoIntersection
        
        d = t * ray - self.to_global(np.array([self.width / 2, h]))
        sign = np.sign(d @ self.rotation[:, 0])

        return np.array([self.width / 2 + sign * theta * self.radius, h])

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
        return self.rotation @ local + self.shift
