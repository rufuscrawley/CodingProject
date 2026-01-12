import numpy as np

natural_units = True
G = 1 if natural_units else 6.67E-11


def vector_sq_mag(v):
    result = 0
    result += np.pow(v.x, 2)
    result += np.pow(v.y, 2)
    return result


class Vector2D:
    """
    Helper class to store a 2D vector. Features simple implementations of
    any relevant vector operations.
    """
    x = 0.0
    y = 0.0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def multiply(self, scalar):
        self.x *= scalar
        self.y *= scalar

    def add(self, scalar):
        self.x += scalar
        self.y += scalar

    def add_mult(self, scalar):
        self.x += self.x * scalar
        self.y += self.y * scalar

    def sq_mag(self):
        result = 0
        result += np.pow(self.x, 2)
        result += np.pow(self.y, 2)
        return result


class Body:
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
            dampener = 0.1 ** 2
            dist: float = np.sqrt(sq_dist)

            magnitude: float = (G * body.mass) / (sq_dist + dampener)
            acc.x += magnitude * (body.pos.x - self.pos.x) / dist
            acc.y += magnitude * (body.pos.y - self.pos.y) / dist
        self.acc = acc
        print(f"s ({self.pos.x}, {self.pos.y})")
        print(f"v ({self.vel.x}, {self.vel.y})")
        print(f"a ({acc.x}, {acc.y})")

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


def get_time_input():
    end = input("Please input the desired running time of the simulation in seconds: ")
    step = input("Please input the desired time-step of the simulation in seconds: ")
    return float(end), float(step)
