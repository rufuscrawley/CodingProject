from copy import deepcopy
from typing import Any

import numpy as np
from alive_progress import alive_bar

from Body import Body
from utilities import get_decimal_places, centre_of_mass


class VerletOutput(object):
    """
    Helper class for handling the Verlet outputs.
    """
    def __init__(self, bodies: list, energies: list[list], ams: list) -> None:
        self.bodies = bodies
        self.energies = energies
        self.ams = ams


def verlet_integration(bodies: list[Body], end: float, step: float,
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
