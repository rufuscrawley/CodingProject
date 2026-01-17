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
        self.energies = energies  # should be three lists for KE, GPE, and total E
        self.ams = ams


def integration(bodies: list[Body], end: float, step: float,
                three_body: bool, softener: float) -> VerletOutput:
    """
    Performs Verlet integration over our system.
    :param bodies: A list of bodies over which we iterate.
    :param end: The total time the integration will run for
    :param step: The timestep over which we integrate
    :param three_body: Whether we are integrating the three-body solutions (and thus need to equate G = 1)
    :param softener: The softener parameter to apply when bodies are close
    :return: An array of the Body values involved
    """
    # Time to 0.
    t: float = 0.0
    iterations: int = 0
    total_steps = int(end / step)
    halfstep = step / 2
    print(f"Beginning Verlet integration for {len(bodies)} bodies over {total_steps} steps.")
    output = VerletOutput([], [[], [], []], [])
    # Use a progress bar (implemented purely for any 30 minute simulations!)
    with alive_bar(total_steps) as bar:
        while (t + step) <= end:
            # Calculate the position of the centre of mass first.
            pos_cm = centre_of_mass(bodies)
            for body in bodies:
                if iterations == 0:
                    # For the first run, calculate all initial conditions.
                    E_k_0 = body.ke()
                    E_p_0 = body.gpe(bodies, three_body, softener)
                    E_t_0 = E_k_0 + (E_p_0 / 2)
                    L_0 = body.am(pos_cm)
                    # Now attach to all the lists.
                    output.bodies.append(deepcopy(body))
                    output.energies[0].append(E_k_0)
                    output.energies[1].append(E_p_0)
                    output.energies[2].append(E_t_0)
                    output.ams.append(deepcopy(L_0))
                    # Move to the next loop
                    continue
                # First calculate the half-step velocities
                body.accelerate(bodies, three_body, softener)
                body.vel.x.add(body.acc.x * halfstep)
                body.vel.y.add(body.acc.y * halfstep)
                # Next, recalculate our position
                body.pos.x.add(body.vel.x * step)
                body.pos.y.add(body.vel.y * step)
                # Then more half-step velocities
                body.accelerate(bodies, three_body, softener)
                body.vel.x.add(body.acc.x * halfstep)
                body.vel.y.add(body.acc.y * halfstep)
                # Now append to a list we can output
                # We need to use deepcopy to not use "static" instance
                # of the body!
                output.bodies.append(deepcopy(body))
                ke = body.ke()
                gpe = body.gpe(bodies, three_body, softener)
                am = body.am(pos_cm)
                output.energies[0].append(ke)
                output.energies[1].append(gpe)
                output.energies[2].append(ke + (gpe / 2))
                output.ams.append(am)
            t += step
            iterations += 1
            if np.log10(step) < 1:
                t = np.round(t, get_decimal_places(step) + 1)
            bar()
    return output


def split_list(target_list: list, offset: int, split: int) -> list:
    """
    Splits a list into a new single list that picks from every N values [from the original list].
    :param target_list: The list to apply this operation to.
    :param offset: The index to start at when splitting.
    :param split: The number of values to split into.
    :return:
    """
    # Create a new separate list
    return_list = target_list.copy()
    # If offset is 0, we don't need to pop our original values
    if offset != 0:
        for i in range(offset):
            return_list.pop(0)
    # Now grab the remaining values
    return return_list[::split]


def get_decimal_places(value: float) -> int:
    """
    Returns the number of decimal places of a floating point number.
    Used to round numbers to mitigate floating-point error.
    :param value: The number of which to get the decimal place value.
    """
    return int(np.round(-np.log10(value)))


def centre_of_mass(bodies: list[Body]) -> Vector2D:
    """
    Calculates the centre of mass of a list of bodies.
    :return: Position and then velocity as a Tuple.
    :param bodies: The bodies to calculate CoM of.
    """
    # Define initial conditions of our system
    total_mass = 0
    pos = Vector2D(0, 0)
    # Multiply over each body
    for body in bodies:
        total_mass += body.mass
        pos.x.add(body.mass * body.pos.x)
        pos.y.add(body.mass * body.pos.y)
    # Normalise CoM
    pos.multiply(1 / total_mass)
    return pos


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


def line_text() -> None:
    """
    Again, just draws a divider line. Useful for my own CLI debugging.
    :return:
    """
    print("=================================")
