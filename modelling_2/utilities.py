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


def centre_of_mass(bodies: list[Body]) -> Vector2D:
    """
    Calculates the centre of mass of a list of bodies.
    :return: Position and then velocity as a Tuple.
    :param bodies: The bodies to calculate CoM of.
    """
    # Define initial conditions of our system
    total_mass = 0
    pos_com = Vector2D(0, 0)
    # Multiply over each body
    for body in bodies:
        total_mass += body.mass

        pos_com.x += body.mass * body.pos.x
        pos_com.y += body.mass * body.pos.y

    # Normalise CoM
    pos_com.multiply(1 / total_mass)

    return pos_com


def get_total_var(body_list: list, body_n: int) -> list:
    """
    Finds the total of a variable of a list.
    :param body_n: Number of bodies in the list
    :param body_list: The list to handle
    :return: Total sum of the variable.
    """
    params = []
    final_vars = []
    # Make a list of each
    for i in range(body_n):
        params.append(split_list(body_list, i, body_n))
    for i in range(len(params[0])):
        var = 0
        for j in range(body_n):
            var += params[j][i]
        final_vars.append(var)
    return final_vars


def is_within_percentage(n_1: float, n_2: float, percentage: float) -> bool:
    """
    Returns whether one number is within a % value of another number.
    :param n_1:
    :param n_2:
    :param percentage: A percentage, out of 100.
    :return: 1 if the conditions are met.
    """
    decimal = percentage / 200.0
    top = n_2 * (1.0 + decimal)
    bottom = n_2 * (1.0 - decimal)
    return (bottom <= n_1) and (n_1 <= top)


def cool_text() -> None:
    """
    Just a flashy function to add a cool title.
    :return: A cool title!
    """
    print("=================================")
    print("=  __      __       _      _    =\n"
          "=  \\ \\    / /      | |    | |   =\n"
          "=   \\ \\  / /__ _ __| | ___| |_  =\n"
          "=    \\ \\/ / _ \\ '__| |/ _ \\ __| =\n"
          "=     \\  /  __/ |  | |  __/ |_  =\n"
          "=      \\/ \\___|_|  |_|\\___|\\__| =")
    print("=================================")
