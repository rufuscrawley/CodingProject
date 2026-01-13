from copy import deepcopy
from typing import Any

import alive_progress
import numpy as np
from alive_progress import alive_bar

from Body import Body
from utilities import get_decimal_places, centre_of_mass


def verlet_integration(bodies: list[Body], end: float, step: float,
                       natural: bool) -> tuple[list[Any], list[Any], list[Any]]:
    """
    Performs Verlet integration over our system.
    :return: An array of the Body values involved
    """
    # Time to 0.
    t: float = 0
    iterations: int = 0
    total_steps = int(end / step)
    print(f"Beginning Verlet integration for {len(bodies)} bodies over {total_steps} steps.")
    body_list, energy_list, am_list = [], [], []
    with alive_bar(total_steps) as bar:
        while (t + step) <= end:
            # Calculate the position and velocity of the centre of mass first.
            pos_cm = centre_of_mass(bodies)
            for body in bodies:
                if iterations == 0:
                    E_k_0: float = body.ke()
                    E_p_0: float = body.gpe(bodies, natural)
                    E_t_0: float = E_k_0 - E_p_0
                    L_0: float = body.am(pos_cm)
                    body_list.append(deepcopy(body))
                    energy_list.append(E_t_0)
                    am_list.append(L_0)
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
                body_list.append(deepcopy(body))
                energy_list.append(deepcopy(body.ke() - body.gpe(bodies, natural)))
                am_list.append(deepcopy(body.am(pos_cm)))
            t += step
            iterations += 1
            if np.log10(step) < 1:
                t = np.round(t, get_decimal_places(step))
            bar()

    return body_list, energy_list, am_list
