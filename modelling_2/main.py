import matplotlib.pyplot as plt
import numpy as np

from Body import Body, setup_bodies
from maths import verlet_integration
from utilities import split_list


# region graphing
def graph_positions(arguments: list[Body], body_count: int, axes):
    x_list = list(map(lambda argument: argument.pos.x, arguments))
    y_list = list(map(lambda argument: argument.pos.y, arguments))

    for i in range(body_count):
        axes.plot(split_list(x_list, i, body_count),
                  split_list(y_list, i, body_count),
                  label=arguments[i].name)
    axes.set_title("Positions")


def graph_total_energies(arguments, body_count: int,
                         end: int, step: int, axes):
    for i in range(body_count):
        axes.plot(np.arange(0, end, step).tolist(),
                  split_list(arguments[1], i, body_count),
                  label=arguments[0][i].name)
    axes.set_title("Total energy")


def graph_am(arguments, body_count: int, end: int, step: int, axes):
    for i in range(body_count):
        axes.plot(np.arange(0, end, step).tolist(),
                  split_list(arguments[2], i, body_count),
                  label=arguments[0][i].name)
    axes.set_title("Angular Momentum")


# endregion

# Read in the bodies that we are working with.
bodies: list[Body] = setup_bodies("csvs/figure_eight.csv")
body_count = len(bodies)
# Set the step count for our Verlet integration.
end, step = 20, 1
# Run the Verlet integration.
arguments = verlet_integration(bodies, end, step)

# Now let's get graphing
fig, axs = plt.subplots(2, 2)
graph_positions(arguments[0], body_count, axs[0, 0])
graph_total_energies(arguments, body_count, end, step, axs[0, 1])
graph_am(arguments, body_count, end, step, axs[1, 0])
plt.show()
