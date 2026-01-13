from copy import deepcopy

import numpy as np
from alive_progress import alive_bar

from Body import Body
from Vector2D import Vector2D


class VerletOutput(object):
    """
    Helper class for handling the Verlet outputs.
    """

    def __init__(self, bodies: list, energies: list[list], ams: list) -> None:
        self.bodies = bodies
        self.energies = energies
        self.ams = ams


def integration(bodies: list[Body], end: float, step: float,
                natural: bool) -> VerletOutput:
    """
    Performs Verlet integration over our system.
    :return: An array of the Body values involved
    """
    # Time to 0.
    t: float = 0
    iterations: int = 0
    total_steps = int(end / step)
    print(f"Beginning Verlet integration for {len(bodies)} bodies over {total_steps} steps.")
    output = VerletOutput([], [[], [], []], [])
    with alive_bar(total_steps) as bar:
        while (t + step) <= end:
            # Calculate the position and velocity of the centre of mass first.
            pos_cm = centre_of_mass(bodies)
            for body in bodies:
                if iterations == 0:
                    # For the first run, calculate all initial conditions.
                    E_k_0 = body.ke()
                    E_p_0 = body.gpe(bodies, natural)
                    E_t_0 = E_k_0 + E_p_0
                    L_0 = body.am(pos_cm)
                    # Now attach to all the lists.
                    output.bodies.append(deepcopy(body))
                    output.energies[0].append(E_k_0)
                    output.energies[1].append(E_p_0)
                    output.energies[2].append(E_t_0)
                    output.ams.append(L_0)
                    # Move to the next loop
                    continue
                # First calculate the half-step velocities
                body.update_acceleration(bodies, natural)
                body.vel.x += (body.acc.x * (step / 2))
                body.vel.y += (body.acc.y * (step / 2))
                # Next, recalculate our position
                body.pos.x += (body.vel.x * step)
                body.pos.y += (body.vel.y * step)
                # Then more half-step velocities
                body.update_acceleration(bodies, natural)
                body.vel.x += (body.acc.x * (step / 2))
                body.vel.y += (body.acc.y * (step / 2))
                # Now append to a list we can output
                # We need to use deepcopy to not use "static" instance
                # of the body!
                output.bodies.append(deepcopy(body))
                ke = body.ke()
                gpe = body.gpe(bodies, natural)
                output.energies[0].append(ke)
                output.energies[1].append(gpe)
                output.energies[2].append(ke + gpe)
                output.ams.append((body.am(pos_cm)))
            t += step
            iterations += 1
            if np.log10(step) < 1:
                t = np.round(t, get_decimal_places(step))
            bar()
    return output


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
    :param n_1: The number we are comparing against
    :param n_2: The number we are going to compare
    :param percentage: A percentage, out of 100.
    :return: 1 if the conditions are met.
    """
    decimal = percentage / 200.0
    top = n_2 * (1.0 + decimal)
    bottom = n_2 * (1.0 - decimal)
    if n_2 <= 0:
        return (top <= n_1) and (n_1 <= bottom)
    else:
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
