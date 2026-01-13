import statistics

import matplotlib.pyplot as plt

from Body import setup_bodies
from utilities import *


def main():
    cool_text()
    # Read in the bodies that we are working with.
    bodies: list[Body] = setup_bodies("csvs/earth.csv")
    body_count = len(bodies)
    # Set the step count for our Verlet integration.
    end, step = 35_000_000, 1000
    # Run the Verlet integration.
    verlet = integration(bodies, end, step, False)

    # Now let's get graphing
    print("Beginning graphing...")

    # Plot positions
    # noinspection PyUnusedLocal
    pos_plot = plt.figure(1)
    x_list = list(map(lambda argument: float(argument.pos.x), verlet.bodies))
    y_list = list(map(lambda argument: float(argument.pos.y), verlet.bodies))
    for i in range(body_count):
        plt.plot(split_list(x_list, i, body_count),
                 split_list(y_list, i, body_count),
                 label=verlet.bodies[i].name)
    plt.title("Positions")

    # Plot energies
    energy_plot, ax = plt.subplots(2, 2)
    ke = get_total_var(verlet.energies[0], body_count)
    gpe = get_total_var(verlet.energies[1], body_count)
    e_t = get_total_var(verlet.energies[2], body_count)
    ax[0, 0].plot(np.arange(0, end, step).tolist(), ke)
    ax[0, 0].set_title("ke")
    ax[1, 0].plot(np.arange(0, end, step).tolist(), gpe)
    ax[1, 0].set_title("gpe")
    ax[0, 1].plot(np.arange(0, end, step).tolist(), e_t)
    ax[0, 1].set_title("e_t")
    ax[1, 1].plot(np.arange(0, end, step).tolist(),
                  list(map(lambda e: (e / (verlet.energies[2][0] + verlet.energies[2][1])), e_t)))
    ax[1, 1].set_title("de_t")

    # Plot angular momentum
    # noinspection PyUnusedLocal
    am_plot = plt.figure(3)
    ams = get_total_var(verlet.ams, body_count)
    plt.plot(np.arange(0, end, step).tolist(), ams)
    plt.title("Angular Momentum")
    plt.show()

    # Guess period
    e_0 = e_t.pop(0)
    correct_times = []
    starting_percentage = 1E-9
    guessed_periods = 0
    while guessed_periods < 2:
        for n, energy in enumerate(e_t):
            if (is_within_percentage(energy, e_0, starting_percentage)
                    and (n > int(len(e_t) * 0.05))):
                correct_times.append(n * step)
                guessed_periods += 1
        if starting_percentage > 100:
            print("Could not guess!")
            break
        print("mult time")
        starting_percentage *= 10
    if correct_times:
        mean = statistics.fmean(correct_times)
        error = ((mean * starting_percentage) / np.sqrt(len(correct_times))) + step
        print("=================================")
        print(f"-> Period: {mean} seconds (+/- {np.round(error, 3)})")
        print(f"-> (That's {mean / 86400} days (+/- {np.round(error / 86400, 3)}))")
        print(f"% uncertainty: {float(error * 100 / mean)}")
        print("=================================")


main()
