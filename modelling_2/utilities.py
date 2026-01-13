import numpy as np

from Body import Body
from Vector2D import Vector2D


def get_time_input():
    end = input("Please input the desired running time of the simulation in seconds: ")
    step = input("Please input the desired time-step of the simulation in seconds: ")
    return float(end), float(step)


def split_list(target_list: list, offset: int, split: int) -> list:
    return_list = target_list.copy()
    if offset != 0:
        for i in range(offset):
            return_list.pop(0)
    return return_list[::split]


def get_decimal_places(value):
    return int(np.round(-np.log10(value)))


def centre_of_mass(bodies: list[Body]) -> tuple[Vector2D, Vector2D]:
    """
    Calculates the centre of mass of a list of bodies.
    :return: Position and then velocity as a Tuple.
    :param bodies: The bodies to calculate CoM of.
    """
    # Define initial conditions of our system
    total_mass = 0
    pos_com = Vector2D(0, 0)
    pos_vel = Vector2D(0, 0)
    # Multiply over each body
    for body in bodies:
        total_mass += body.mass
        pos_com.add_mult(body.mass)
        pos_vel.add_mult(body.mass)
    # Normalise CoM
    pos_com.multiply(1 / total_mass)
    pos_vel.multiply(1 / total_mass)

    return pos_com, pos_vel


def cool_text():
    print("==============================")
    print(" __      __       _      _\n"
          " \\ \\    / /      | |    | |  \n"
          "  \\ \\  / /__ _ __| | ___| |_\n"
          "   \\ \\/ / _ \\ '__| |/ _ \\ __|\n"
          "    \\  /  __/ |  | |  __/ |_\n"
          "     \\/ \\___|_|  |_|\\___|\\__|")
    print("==============================")
