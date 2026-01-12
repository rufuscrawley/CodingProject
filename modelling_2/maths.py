from copy import deepcopy
from typing import Any

import numpy as np

from Body import Body
from Vector2D import Vector2D


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


def verlet_integration(bodies: list[Body], end: float,
                       step: float) -> tuple[list[Any], list[Any], list[Any]]:
    """
    Performs Verlet integration over our system.
    :return: An array of the Body values involved
    """
    print("Running Verlet simulation!")
    print(f"[Max Time] {end}s | [Timestep] {step}s | [Steps] {np.floor(end / step)}")
    # Time to 0.
    t: float = 0
    # Allocate empty lists to store the x/y positions.
    body_list, energy_list, am_list = [], [], []
    body_list.append(deepcopy(bodies[0]))
    body_list.append(deepcopy(bodies[1]))
    body_list.append(deepcopy(bodies[2]))
    while (t + step) <= end:
        # Calculate the position and velocity of the centre of mass first.
        pos_cm, vel_cm = centre_of_mass(bodies)
        for body in bodies:
            # First calculate the half-step velocities
            body.update_acceleration(bodies)
            body.vel.x += (body.acc.x * (step / 2))
            body.vel.y += (body.acc.y * (step / 2))
            # Next, recalculate our position
            body.pos.x += (body.vel.x * step)
            body.pos.y += (body.vel.y * step)
            # Then more half-step velocities
            body.update_acceleration(bodies)
            body.vel.x += (body.acc.x * (step / 2))
            body.vel.y += (body.acc.y * (step / 2))
            # Now calculate relevant parameters
            E_k: float = body.ke()
            E_p: float = body.gpe(bodies)
            E_t: float = E_k + E_p
            L: float = body.am(pos_cm)
            # Now append to a list we can output
            body_list.append(deepcopy(body))
            energy_list.append(deepcopy(E_t))
            am_list.append(deepcopy(L))
        t += step

    print("Verlet Integration finished successfully!")

    return body_list, energy_list, am_list
