import matplotlib.pyplot as plt
import numpy as np

from Body import Body, setup_bodies
from utilities import split_list, cool_text
from verlet import verlet_integration


# region graphing
def graph_positions(body_list: list[Body], body_n: int):
    pos_plot = plt.figure(1)
    x_list = list(map(lambda argument: float(argument.pos.x), body_list))
    y_list = list(map(lambda argument: float(argument.pos.y), body_list))
    for i in range(body_n):
        plt.plot(split_list(x_list, i, body_n),
                 split_list(y_list, i, body_n),
                 label=body_list[i].name)
    plt.title("Positions")


def graph_total_energies(body_list, body_n: int,
                         end: float, step: float):
    energy_plot = plt.figure(2)
    energies = get_total_var(body_n, body_list, 1)
    plt.plot(np.arange(0, end, step).tolist(), energies)
    plt.title("Total energy")


def graph_am(body_list, body_n: int, end: float, step: float):
    am_plot = plt.figure(3)
    ams = get_total_var(body_n, body_list, 2)
    plt.plot(np.arange(0, end, step).tolist(), ams)
    plt.title("Angular Momentum")


def get_total_var(body_n, body_list, list_index):
    params = []
    final_vars = []
    for i in range(body_n):
        params.append(split_list(body_list[list_index], i, body_n))
    for i in range(len(params[0])):
        var = 0
        for j in range(body_n):
            var += params[j][i]
        final_vars.append(var)
    return final_vars


# endregion

cool_text()
# Read in the bodies that we are working with.
bodies: list[Body] = setup_bodies("csvs/two_body.csv")
body_count = len(bodies)
# Set the step count for our Verlet integration.
end, step = 35_000_000, 1000
# Run the Verlet integration.
arguments = verlet_integration(bodies, end, step, False)
# Now let's get graphing
print("Beginning graphing...")
graph_positions(arguments[0], body_count)
graph_total_energies(arguments, body_count, end, step)
graph_am(arguments, body_count, end, step)
plt.show()
