import statistics

import matplotlib.pyplot as plt

from Body import setup_bodies
from utilities import *
import matplotlib.pylab as pylab

# Change the global graphing parameters
params = {'legend.fontsize': 'x-small',
          'axes.labelsize': 'x-small',
          'axes.titlesize': 'small',
          'xtick.labelsize': 'x-small',
          'ytick.labelsize': 'x-small'}
pylab.rcParams.update(params)


def main():
    cool_text()
    # Read in the bodies that we are working with.
    bodies: list[Body] = setup_bodies("csvs/earth.csv")
    body_count = len(bodies)
    # Set the step count for our Verlet integration.
    end, step = 50_000_000, 10_000
    # Run the Verlet integration.
    verlet = integration(bodies, end, step, False)

    # Guess period
    e_t = get_total_var(verlet.energies[2], body_count)
    e_0 = e_t.pop(0)
    correct_times = []
    starting_percentage = 1E-5 / 100
    mean = 0
    while not correct_times:
        for n, energy in enumerate(e_t):
            if (is_within_percentage(energy, e_0, starting_percentage)
                    and (n > int(len(e_t) * 0.01))):
                correct_times.append(n * step)
                print(f"{energy} is within {starting_percentage}% of {e_0}")
                print(f"for {correct_times}")
                break
        if starting_percentage > 1:
            print("Could not guess!")
            break
        print("mult time")
        if not correct_times:
            starting_percentage *= 10
        else:
            break
    if correct_times:
        mean = statistics.fmean(correct_times)
        error = ((mean * starting_percentage) / np.sqrt(len(correct_times))) + step
        print(f"error = {mean} * {starting_percentage} / {np.sqrt(len(correct_times))} + {step}")
        print("=================================")
        print(f"-> Predicted period: {mean} seconds (+/- {np.round(error, 3)})")
        print(f"-> (That's {mean / 86400} days (+/- {np.round(error / 86400, 3)}))")
        print(f"% uncertainty: {float(error * 100 / mean)}")
        print("=================================")
    e_t.insert(0, e_0)

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
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")

    # Plot energies
    energy_plot, ax = plt.subplots(2)
    ax[0].plot(np.arange(0, end, step).tolist(), e_t)
    de = list(map(lambda e: (e * 100 / e_0), e_t))
    if correct_times and mean:
        ax[0].vlines(mean, np.max(e_t), np.min(e_t), linestyles="dashed")
        ax[1].vlines(mean, np.max(de), np.min(de), linestyles="dashed")
    ax[0].set_ylabel("Total energy (J)")
    ax[1].plot(np.arange(0, end, step).tolist(),
               de)
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("% of total energy")

    # Plot angular momentum
    # noinspection PyUnusedLocal
    am_plot = plt.figure(3)
    ams = get_total_var(verlet.ams, body_count)
    if correct_times and mean:
        plt.vlines(mean, np.max(ams), np.min(ams), linestyles="dashed")
    plt.plot(np.arange(0, end, step).tolist(), ams)
    plt.ylim(np.max(ams) * 0.5, np.max(ams) * 2)
    plt.xlabel("Time (s)")
    plt.ylabel("Angular momentum (kgm^2s^-1)")
    plt.show()


main()
