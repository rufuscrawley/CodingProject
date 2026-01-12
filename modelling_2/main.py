import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from maths import verlet_integration
from utilities import Body, Vector2D


def setup_bodies(filename: str) -> list[Body]:
    """
    Builds a list of Body objects from a .csv file.
    :param filename: Desired filename to read bodies from.
    :return: List of correctly formatted bodies.
    """
    body_list = []
    # Read in our bodies from the file
    lines: pd.DataFrame = pd.read_csv(filename)
    # Iterate over each body and create a new Body object for each one
    for i in range(len(lines)):
        body = Body(lines.get("name")[i], lines.get("mass")[i],
                    Vector2D(lines.get("pos_x")[i], lines.get("pos_y")[i]),
                    Vector2D(lines.get("vel_x")[i], lines.get("vel_y")[i]))
        body_list.append(body)
    return body_list


def split_list(target_list: list, offset: int, split: int) -> list:
    return_list = target_list.copy()
    if offset != 0:
        for i in range(offset):
            return_list.pop(0)
    return return_list[::split]


def main() -> list[Body]:
    """
    Runs the main body of the code. Returns a list of Bodies.
    Listed as returning type list[Any] to suppress IDE errors.
    :return: A list of bodies.
    """
    # Read in the bodies that we are working with.
    bodies: list[Body] = setup_bodies("butterfly.csv")
    body_count = len(bodies)
    # Set the step count for our Verlet integration.
    end, step = 120, 1
    # Run the Verlet integration.
    arguments = verlet_integration(bodies, end, step)

    # Now let's get graphing
    fig, axs = plt.subplots(2, 2)
    graph_positions(arguments[0], body_count, axs[0, 0])
    graph_total_energies(arguments, body_count, end, step, axs[0, 1])
    graph_am(arguments, body_count, end, step, axs[1, 0])
    plt.legend()
    plt.show()


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


main()
