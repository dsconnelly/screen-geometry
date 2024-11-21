from typing import Optional

import numpy as np

from .utils import get_angle

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
            Center of the screen, in global coordinates. In your implementation,
            if the eye is to be the center of the global coordinate system,
            you might add a routine to update the shift parameter of the screen.
            The rotation matrices would NOT need to be updated, but the
            self.base attribute for curved screens would need to be recalculated
            whenever the eye moves.

            Also note that shift is from the eye to the center of the screen.
            If in practice you are obtaining the eye position from the camera,
            you might be getting something like the negative of this vector
            (from the screen to the eye) and need to adjust accordingly.
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

        # Rotation is a matrix whose columns give a rotated basis corresponding
        # to the directions at the center of the screen. The first column goes
        # width-wise, the second column points height-wise, and the third points
        # perpendicular outward. I store that one as the "normal", although note
        # that for curved screens this is only the normal at the center.

        self.rotation = yaw_matrix @ pitch_matrix
        self.normal = self.rotation[:, 2]

        if self.radius is not None:
            # Here I store self.base as the 3D coordinates of a point at the
            # center of the cylinder, at the height of the bottom of the screen
            # (i.e. the center of the cylindrical "end cap"). Then axis is a
            # unit vector pointing along the direction of the cylinder.

            p = self.to_global(np.array([self.width / 2, 0]))
            self.base = p + self.radius * self.normal
            self.axis = self.rotation[:, 1]

            # theta_max is the angular displacement of the end of the screen
            # from its center. For example, if the screen were semicircular,
            # theta_max would be 90 degrees (stored here in radians).

            q = self.to_global(np.array([self.width, 0]))
            self.theta_max = get_angle(p - self.base, q - self.base)

    def find_intersect(self, ray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
            Array of global coordinates at the point of intersection.
        np.ndarray
            Array of pixel coordinates at the point of intersection.

        """

        if self.radius is None:
            # Flat screen, so we use the line-plane intersection formula. Here
            # t is the coefficient along the ray to reach the intersection -- if
            # ray is a unit vector, then t corresponds to distance.
            t = (self.shift @ self.normal) / (ray @ self.normal)

            if t < 0:
                # The screen is behind us.
                raise NoIntersection
            
            # The 3D coords of the intersection point are just t * ray, but to
            # get into pixel coordinates, it's temporarily useful to remove the
            # shift of the screen away from the origin.
            point = t * ray - self.shift

            # In pixel coordinates, the x and y coordinates are just the dot
            # product of the displacement from the center with the local basis
            # vectors (first two columns of rotation). Then I add width / 2 and
            # height / 2 because the position vector was relative to the center.
            x = self.rotation[:, 0] @ point + self.width / 2
            y = self.rotation[:, 1] @ point + self.height / 2

            if not (0 <= x <= self.width and 0 <= y <= self.height):
                # The ray intersects the plane, but not the portion of the plane
                # actually occupied by the screen.
                raise NoIntersection
            
            # Before returning, we add back in the position of the screen.
            return point + self.shift, np.array([x, y])
        
        # Curved screen, so we use the line-cyilnder intersection formula. In
        # the Wikipedia article, their "n" corresponds to my "ray". I use "rxa"
        # to store the ray x self.axis cross product, as it appears several
        # times in the formula.

        rxa = np.cross(ray, self.axis)
        sq = np.sqrt(self.radius ** 2 * (rxa @ rxa) - (self.base @ rxa) ** 2)
        t = (rxa @ np.cross(self.base, self.axis) + sq) / (rxa @ rxa)

        # t has the same interpretation as before. Now we have the point of
        # intersection; what's left is bounds checking for validity and
        # conversion to pixel coordinates.
        point = t * ray

        # The projection of (point - base) onto the axis vector gives us the
        # displacement of the screen height.
        y = self.axis @ (point - self.base)

        if not 0 <= y <= self.height:
            # The line intersects the infinite cylinder, but not in the finite
            # range of heights where the screen actually is.
            raise NoIntersection

        # Here "center" is the 3D coords of a point at the center of the
        # cylinder but at the same screen height as the intersection point.
        center = self.base + y * self.rotation[:, 1]

        # q is a point at the center of the screen width-wise, and at the same
        # screen height as the intersection point.
        q = self.to_global(np.array([self.width / 2, y]))

        # Therefore point - center and q - center are vectors from the central
        # axis to the intersection point and the center of the screen. So the
        # angle between them tells as how far along the circumference of the
        # circle the point of intersection is from the center.
        theta = get_angle(point - center, q - center)

        if theta > self.theta_max:
            # The ray intersects the cylinder at an appropriate height, but not
            # within the sector of the circular wall where the screen is.
            raise NoIntersection

        # The width coordinate in pixels of the intersection point will be
        # theta * radius (the length of the circular arc) but which direction?
        # To find out, we project the vector from the center onto the local
        # x basis vector (pointing width-wise) and check the sign.
        sign = np.sign((point - q) @ self.rotation[:, 0])

        # Now we just add back in width / 2 so that the pixel coordinates are
        # relative to the lower left corner.
        x = self.width / 2 + sign * theta * self.radius

        return point, np.array([x, y])

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

        # The provided pixel coordinates are relative to the lower left corner,
        # but the position vector of the screen describes the screen center, so
        # we'd like the pixel coordinates relative to the screen center.
        x_shift = pixels[0] - self.width / 2
        y_shift = pixels[1] - self.height / 2
        
        # We have different behavior for curved and flat screens, except in the
        # y direction, since even a curved screen is "flat" height-wise.

        if self.radius is None:
            # The screen is flat, so all points are at z = 0 in the local
            # (rotated) coordinate system, and the coordinate in the local
            # x direction is just the shifted pixel coordinate.
            x_local = x_shift
            z_local = 0

        else:
            # The screen is curved. First, we calculate the angular displacement
            # of the point around the screen from the center.
            angle = x_shift / self.radius

            # Now, we use trigonometry to find the displacement from the screen
            # center in the local (rotated) coordinate system.
            x_local = self.radius * np.sin(angle)
            z_local = self.radius * (1 - np.cos(angle))

        # Now we have the position of the point in 3D space, but expressed
        # relative to the center of the screen using the rotated basis.
        abc = np.array([x_local, y_shift, z_local])

        # We have expressed our point as
        #     p = shift + a * i_local + b * j_local + c * k_local
        # where i_local, j_local, and k_local are the rotated basis vectors
        # pointing width-wise, height-wise, and perpendicular to the center of
        # the screen (e.g. the columns of self.rotation). But
        #     i_local = rotation @ i
        # where i is the first standard basis vector, and likewise for j_local
        # and k_local. So we have
        #     p = shift + rotation @ (a * i + b * j + c * k)
        # and so if we interpret the abc coefficients as corresponding to the
        # standard basis, we get the coordinates in the global system like so.
        return self.rotation @ abc + self.shift
