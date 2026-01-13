import statistics

import matplotlib.pyplot as plt

from Body import setup_bodies
from utilities import *
from verlet import verlet_integration

cool_text()
# Read in the bodies that we are working with.
bodies: list[Body] = setup_bodies("csvs/two_body.csv")
body_count = len(bodies)
# Set the step count for our Verlet integration.
end, step = 500_000_000, 1_000
# Run the Verlet integration.
arguments = verlet_integration(bodies, end, step, False)
# Now let's get graphing
print("Beginning graphing...")

# Plot positions
pos_plot = plt.figure(1)
x_list = list(map(lambda argument: float(argument.pos.x), arguments[0]))
y_list = list(map(lambda argument: float(argument.pos.y), arguments[0]))
for i in range(body_count):
    plt.plot(split_list(x_list, i, body_count),
             split_list(y_list, i, body_count),
             label=arguments[0][i].name)
plt.title("Positions")
# Plot energies
energy_plot = plt.figure(2)
energies = get_total_var(body_count, arguments, 1)
plt.plot(np.arange(0, end, step).tolist(), energies)
plt.title("Total energy")
# Plot angular momentum
am_plot = plt.figure(3)
ams = get_total_var(body_count, arguments, 2)
plt.plot(np.arange(0, end, step).tolist(), ams)
plt.title("Angular Momentum")
plt.show()
# Guess period
e_0 = energies.pop(0)
correct_times = []
starting_percentage = 1E-9
guessed_periods = 0
while guessed_periods < 2:
    for n, energy in enumerate(energies):
        if (is_within_percentage(energy, e_0, starting_percentage)
                and (energy != 0) and (n > int(len(energies) * 0.05))):
            correct_times.append(n * step)
            guessed_periods += 1
    if starting_percentage > 1:
        print("Could not guess!")
        break
    starting_percentage *= 10

if correct_times:
    mean = statistics.fmean(correct_times)
    error = ((mean * starting_percentage) / np.sqrt(len(correct_times))) + step
    print(f"Guessed period: {mean} seconds (+/- {np.round(error, 3)})")
    print(f"Guessed period: {mean / 86400} days (+/- {np.round(error / 86400, 3)})")
    print(f"% uncertainty: {float(error * 100 / mean)}")
