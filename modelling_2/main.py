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

system_dict = {
    "earth": {
        "end": 50_000_000,
        "step": 1000,
        "natural": False
    },
    "mercury": {
        "end": 8_000_000,
        "step": 1000,
        "natural": False,
    },
    "moon": {
        "end": 2_300_000,
        "step": 100,
        "natural": False,
    },
    "figure_eight": {
        "end": 6.5,
        "step": 0.001,
        "natural": True
    }
}


def main(setup: str):
    name = setup
    end = system_dict[setup]["end"]
    step = system_dict[setup]["step"]
    natural = system_dict[setup]["natural"]

    # Read in the bodies that we are working with.
    bodies: list[Body] = setup_bodies(f"csvs/{name}.csv")
    body_count = len(bodies)
    # Run the Verlet integration.
    verlet = integration(bodies, end, step, natural)

    e_t = get_total_var(verlet.energies[2], body_count)
    # Store initial condition for later
    e_0 = e_t.pop(0)

    # Guess period
    # Start by setting up empty period list + other variables
    periods = []
    guess_percent = 1E-5 / 100
    mean = 0
    # If no successful guesses, guess more within a threshold percentage
    while not periods:
        # Guess over each index n and energy e
        for n, e in enumerate(e_t):
            # Make sure we're not guessing the first 1% of values
            if (is_within_percentage(e, e_0, guess_percent)
                    and (n > int(len(e_t) * 0.01))):
                periods.append(n * step)
                break
        if guess_percent > 1:
            # If we didn't find any over a 100% threshold, things are clearly broken...
            print("Could not guess!")
            break
        if not periods:
            # If we didn't find any, just increase the threshold.
            guess_percent *= 10
        else:
            # And again, break out of the control loop.
            # Could be refactored with a while loop instead?
            break
    if periods:
        # If we found any periods, output the relevant details
        mean = statistics.fmean(periods)
        error = ((mean * guess_percent) / np.sqrt(len(periods))) + step
        print("=================================")
        print(f"-> Predicted period: {mean} seconds (+/- {np.round(error, 3)})")
        print(f"-> (That's {mean / 86400} days (+/- {np.round(error / 86400, 3)}))")
        print(f"% uncertainty: {float(error * 100 / mean)}")
        print("=================================")
    # Put the initial energy back where it belongs.
    e_t.insert(0, e_0)

    # Now let's get graphing
    print("Beginning graphing...")

    # Plot positions
    # noinspection PyUnusedLocal
    pos_plot = plt.figure(1)
    # Create (x, y) lists from our bodies
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
    # Map the change in energy.
    de = list(map(lambda e: (e * 100 / e_0) - 100, e_t))
    if periods and mean:
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
    if periods and mean:
        plt.vlines(mean, np.max(ams), np.min(ams), linestyles="dashed")
    plt.plot(np.arange(0, end, step).tolist(), ams)
    plt.ylim(np.min(ams) * 0.95, np.max(ams) * 1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Angular momentum (kgm^2s^-1)")
    plt.show()

    print("Graphing completed!")
    print("=================================")


# Now begin central control flow
cool_text()
# List out the available system from the dynamic dictionary
print("Systems available:")
for system in system_dict:
    print(f"-> {system}")
print("=================================")
# Ask for the user's input
while True:
    try:
        system = str(input("Please input the desired system.\n"))
        if system in system_dict:
            print(f"System '{system}' found")
            print("=================================")
            break
        else:
            print("Not found in dictionary, please try again")
    except ValueError:
        print("That isn't a string, please try again")
# ... and run the main code.
main(system)
