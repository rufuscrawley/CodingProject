import numpy as np
import pandas as pd

from Vector2D import Vector2D

natural_units = True
G = 1 if natural_units else 6.67E-11


class Body(object):
    name = ""
    mass = 0.0

    pos = Vector2D
    vel = Vector2D
    acc = Vector2D

    def __init__(self, name, mass, pos, vel):
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

    def information(self):
        """
        Prints out useful debug information about the body.
        """
        print(f"========================\n"
              f"- Name: {self.name} - Mass: {self.mass}\n"
              f"- Position: ({self.pos.x}, {self.pos.y})\n"
              f"- Velocity: ({self.vel.x}, {self.vel.y})\n"
              f"========================")

    def update_acceleration(self, bodies: list) -> None:
        """
        Updates the acceleration Vector2D of the body as per a Physics simulation.
        :param bodies: The bodies acting upon this body.
        """
        acc = Vector2D(0, 0)

        for body in bodies:
            if self.name == body.name:
                continue
            sq_dist: float = self.dist_squared(body)

            # dampen
            dampener = 0.01 ** 2
            dist: float = np.sqrt(sq_dist)

            magnitude: float = (G * body.mass) / (sq_dist + dampener)
            acc.x += magnitude * (body.pos.x - self.pos.x) / dist
            acc.y += magnitude * (body.pos.y - self.pos.y) / dist
        self.acc = acc

    def am(self, ref_point: Vector2D) -> float:
        rel_pos = Vector2D(self.pos.x - ref_point.x,
                           self.pos.y - ref_point.y)

        dist: float = np.sqrt(rel_pos.sq_mag())
        vel: float = np.sqrt(self.vel.sq_mag())

        return self.mass * dist * vel

    def dist_squared(self, body):
        dist_vector = Vector2D(self.pos.x - body.pos.x,
                               self.pos.y - body.pos.y)
        return dist_vector.sq_mag()

    def ke(self) -> float:
        return 0.5 * self.mass * self.vel.sq_mag()

    def gpe(self, bodies):
        result = 0
        for body in bodies:
            if self == body:
                continue

            r_squared = self.dist_squared(body)
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
    for i in range(len(lines)):
        body = Body(lines.get("name")[i], lines.get("mass")[i],
                    Vector2D(lines.get("pos_x")[i], lines.get("pos_y")[i]),
                    Vector2D(lines.get("vel_x")[i], lines.get("vel_y")[i]))
        body_list.append(body)
    return body_list
