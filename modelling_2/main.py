import statistics

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

from Body import setup_bodies
from utilities import *

# Change the global graphing parameters
params = {'legend.fontsize': 'x-small',
          'axes.labelsize': 'x-small',
          'axes.titlesize': 'small',
          'xtick.labelsize': 'x-small',
          'ytick.labelsize': 'x-small'}
pylab.rcParams.update(params)

# Could be stored in its own .csv, but I prefer dictionary format for easy additions
system_dict = {
    "earth": {
        "end": 50_000_000,
        "step": 1000,
        "natural": False,
        "info": "This system simulates the Earth orbiting the Sun.",
        "softener": 0.001
    },
    "mercury": {
        "end": 8_000_000,
        "step": 1000,
        "natural": False,
        "info": "This system simulates Mercury orbiting the Sun.",
        "softener": 0.001
    },
    "moon": {
        "end": 2_300_000,
        "step": 10,
        "natural": False,
        "info": "This system simulates the Moon orbiting the Earth.",
        "softener": 0.01
    },
    "three_body": {
        "end": 50_000_000,
        "step": 1_000,
        "natural": False,
        "info": "This system simulates both the Earth and Mercury orbiting the Sun.",
        "softener": 0.0001
    },
    "figure_eight": {
        "end": 6.5,
        "step": 0.001,
        "natural": True,
        "info": "This 3-body system looks like a figure-eight.",
        "softener": 0.0001
    },
    "butterfly": {
        "end": 7.1,
        "step": 0.00001,
        "natural": True,
        "info": "This 3-body system looks like a butterfly.",
        "softener": 0.00001
    },
    "bumblebee": {
        "end": 64.5,
        "step": 0.0001,
        "natural": True,
        "info": "This 3-body system looks like a bumblebee (if you squint).",
        "softener": 0.0001
    },
    "moth": {
        "end": 15.0,
        "step": 0.0001,
        "natural": True,
        "info": "This 3-body system looks like a moth (if you squint).",
        "softener": 0.001
    },
    "goggles": {
        "end": 10.5,
        "step": 0.00001,
        "natural": True,
        "info": "This 3-body system looks like a moth (if you squint).",
        "softener": 0.0001
    },
}


def main(setup: str):
    # Read in all relevant values from our dictionary.
    name = setup
    end = system_dict[setup]["end"]
    step = system_dict[setup]["step"]
    natural = system_dict[setup]["natural"]
    softening_value = system_dict[setup]["softener"]

    # Read in the bodies that we are working with.
    bodies: list[Body] = setup_bodies(f"csvs/{name}.csv")
    body_count = len(bodies)
    # Run the Verlet integration.
    verlet = integration(bodies, end, step, natural, softening_value)

    # Store initial condition for later
    e_t = get_total_var(verlet.energies[2], body_count)
    e_0 = e_t.pop(0)

    # Guess period
    # Could use position for this as well ; less computationally strenous
    # to calculate off energies instead for negligible change
    # Start by setting up empty period list + other variables
    periods = []
    guess_percent = 1E-5 / 100
    period = 0
    # If no successful guesses, guess more within a threshold percentage
    if not natural:
        while not periods:
            # Guess over each index n and energy e
            for n, e in enumerate(e_t):
                # Make sure we're not guessing the first 1% of values
                if (is_within_percentage(e, e_0, guess_percent)
                        and (n > int(len(e_t) * 0.01))):
                    # Sinusoidal shape, so we hit the expected value twice.
                    periods.append(n * step * 2)
                    break
            if guess_percent > 1:
                # If we didn't find any over a 100% threshold, things are clearly broken...
                print("Could not guess!")
                # Should not break rest of code from testing.
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
            period = statistics.fmean(periods)
            error = ((period * guess_percent) / np.sqrt(len(periods))) + step
            line_text()
            print(f"-> Predicted period: {period} seconds (+/- {np.round(error, 3)})")
            print(f"-> (That's {period / 86400} days (+/- {np.round(error / 86400, 3)}))")
            print(f"% uncertainty: {float(error * 100 / period)}")
            line_text()

    # Extract lists of the X- and Y- coordinates from each body.
    x_list = list(map(lambda body: float(body.pos.x), verlet.bodies))
    y_list = list(map(lambda body: float(body.pos.y), verlet.bodies))

    # Now time to validate Kepler's 3rd law
    semi_major_axis: float = 0
    if not natural:
        print("Validating Kepler's 3rd law...")
        # Use Kepler's 3rd law to find a period
        semi_major_axis = np.pow((6.67E-11 * (bodies[0].mass + bodies[1].mass) * (period ** 2))
                                 / (4 * (np.pi ** 2)), 1 / 3)
        # Now calculate the absolute deviation from the predicted semi-major axis from the x- and
        # y- coordinates we just found.
        radius_list = []
        # Only use the orbit of the main object in our CSVs - should be index 1.
        # Please view template.csv for more information.
        central_orbit_x = split_list(x_list, 1, body_count)
        central_orbit_y = split_list(y_list, 1, body_count)
        for n in range(len(central_orbit_x)):
            # Calculate the deviations from the expected semi-major axis for each value
            orbital_radius = np.sqrt((central_orbit_x[n] ** 2) + (central_orbit_y[n] ** 2))
            radius_list.append(100 * ((orbital_radius - semi_major_axis) / semi_major_axis))
        # And then graph the deviation.
        # noinspection PyUnusedLocal
        funny_plot = plt.figure(0)
        plt.plot(np.arange(0, end, step).tolist(), radius_list)
        plt.xlabel("Time (s)")
        plt.ylabel("Deviation from semi-major axis")
        line_text()

    # Put the initial energy back where it belongs.
    e_t.insert(0, e_0)

    # Plot positions
    # noinspection PyUnusedLocal
    pos_plot = plt.figure(1)
    for i in range(body_count):
        plt.plot(split_list(x_list, i, body_count),
                 split_list(y_list, i, body_count),
                 label=verlet.bodies[i].name)
    # Only used if we found a semi-major axis!
    if semi_major_axis != 0:
        axis = plt.Circle((0, 0), semi_major_axis, fill=False, label="Predicted radius")
        plt.gca().add_patch(axis)
    plt.legend()
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")

    # Plot energies
    energy_plot, ax = plt.subplots(2)
    ax[0].plot(np.arange(0, end, step).tolist(), e_t)
    # Map the change in energy.
    d_e = list(map(lambda e_map: (e_map * 100 / e_0) - 100, e_t))
    if periods and period and not natural:
        ax[0].vlines(period, np.max(e_t), np.min(e_t), linestyles="dashed")
        ax[1].vlines(period, np.max(d_e), np.min(d_e), linestyles="dashed")
    ax[0].set_ylabel("Total energy (J)")
    ax[1].plot(np.arange(0, end, step).tolist(), d_e)
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("% of total energy")

    # Plot angular momentum
    # noinspection PyUnusedLocal
    am_plot = plt.figure(3)
    ams = get_total_var(verlet.ams, body_count)
    am_0 = ams[0]
    # Make sure no division by zero - if initial angular momentum is 0, then any deviation
    # can just be treated as ABSOLUTE.
    if am_0 == 0:
        d_ams = ams
    else:
        d_ams = list(map(lambda am: (am * 100 / am_0) - 100, ams))
    plt.plot(np.arange(0, end, step).tolist(), d_ams)
    plt.xlabel("Time (s)")
    plt.ylabel("Change in angular momentum (%) (kgm^2s^-1)")
    plt.show()
    print("Graphing completed!")
    line_text()
    # Additional physical properties of our system
    print("Extra variables:")
    print(f"E_0 = {e_0} J")
    print(f"am_0 = {am_0} angular momentums")
    print(f"a = {semi_major_axis}m")
    line_text()


# Now begin central control flow
cool_text()
# List out the available system from the dynamic dictionary
print("Systems available:")
for system in system_dict:
    print(f"-> {system}")
line_text()
# Ask for the user's input
while True:
    try:
        system = str(input("Please input the desired system.\n"))
        if system in system_dict:
            print(f"System '{system}' found")
            print(system_dict[system]["info"])
            line_text()
            break
        else:
            print("Not found in dictionary, please try again")
    except ValueError:
        print("That isn't a string, please try again")
# ... and run the main code.
main(system)
