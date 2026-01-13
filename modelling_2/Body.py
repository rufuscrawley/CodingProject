import numpy as np
import pandas as pd

from Vector2D import Vector2D


class Body(object):
    name = ""
    mass = 0.0

    pos, vel, acc = Vector2D, Vector2D, Vector2D

    def __init__(self, name: str, mass: float, pos: Vector2D, vel: Vector2D) -> None:
        """
        :param name: The name of the stellar body
        :param mass: The body's mass (kg)
        :param pos: The position of the body (m)
        :param vel: The velocity of the body (ms^-1)
        """
        self.name = name
        self.mass = mass
        self.pos = pos
        self.vel = vel

    def __str__(self):
        # Redefined str() function for debugging values - may not be used in final build.
        return f"[{self.name}] - ({self.pos.x}, {self.pos.y}) @ ({self.vel.x}, {self.vel.y})"

    def update_acceleration(self, bodies: list, natural: bool) -> None:
        """
        Updates the acceleration Vector2D of the body as per a Physics simulation.
        :param natural: Whether to use natural units (G = 1) or not (G = 6.67E-11)
        :param bodies: The bodies acting upon this body.
        """
        # Set up our variables
        acc = Vector2D(0, 0)
        G = 1 if natural else 6.67E-11
        # Now calculate individual accelerations from each body
        for body in bodies:
            if self == body:
                continue
            sq_dist: float = self.dist_squared(body)
            dist: float = np.sqrt(sq_dist)
            # Apply dampening
            dampener = 0.0001 ** 2
            magnitude: float = (G * body.mass) / (sq_dist + dampener)
            # Update the acceleration
            acc.x += magnitude * (body.pos.x - self.pos.x) / dist
            acc.y += magnitude * (body.pos.y - self.pos.y) / dist
        # Apply the final calculated acceleration
        self.acc = acc

    def am(self, ref_point: Vector2D) -> float:
        """
        Calculates the angular momentum of a body against a defined point of reference.

        The point of reference is usually a centre of mass in an N-body problem.
        :param ref_point: The position of the point of reference
        :return: The angular momentum
        """
        # Find the r vector between us and the point of reference.
        rel_pos = Vector2D(self.pos.x - ref_point.x,
                           self.pos.y - ref_point.y)
        # Find distance and velocity vectors.
        dist: float = np.sqrt(rel_pos.sq_mag())
        vel: float = np.sqrt(self.vel.sq_mag())
        # w = m * v * L
        return self.mass * dist * vel

    def dist_squared(self, body) -> float:
        """
        Calculates the distance to a different body.
        :param body: Body to calculate distance to.
        :return: Distance magnitude.
        """
        dist_vector = Vector2D(self.pos.x - body.pos.x,
                               self.pos.y - body.pos.y)
        return dist_vector.sq_mag()

    def ke(self) -> float:
        """
        Calculates the kinetic energy of the body at that moment in time.
        :return: Kinetic energy in [J]
        """
        # E = 1/2 * m * v^2

        energy = 0.5 * self.mass * self.vel.sq_mag()
        return energy

    def gpe(self, bodies: list, natural: bool) -> float:
        """
        Calculates the GPE of the body against nearby other bodies.
        :param bodies: A list of bodies that act upon this body.
        :param natural: Whether to use natural units in the calculation or not.
        :return: The GPE of the body in [J].
        """
        result = 0
        # Should we use natural units?
        G = 1 if natural else 6.67E-11
        for body in bodies:
            if self == body:
                continue
            r_squared = self.dist_squared(body)
            # U = - GMm / r
            result += -1 * (G * self.mass * body.mass) / np.sqrt(r_squared)
        return result


def setup_bodies(filename: str) -> list[Body]:
    """
    Builds a list of Body objects from a .csv file.
    :param filename: Desired filename to read bodies from.
    :return: List of correctly formatted bodies.
    """
    body_list = []
    # Read in our bodies from the file
    lines: pd.DataFrame = pd.read_csv(filename)
    # Iterate over each body and create a new Body object for each one
    print(f"Found {len(lines)} bodies! Loading...")
    for i in range(len(lines)):
        body = Body(lines.get("name")[i], lines.get("mass")[i],
                    Vector2D(lines.get("pos_x")[i], lines.get("pos_y")[i]),
                    Vector2D(lines.get("vel_x")[i], lines.get("vel_y")[i]))
        body_list.append(body)
    print("Bodies loaded!")
    return body_list
