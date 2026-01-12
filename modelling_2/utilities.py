def get_time_input():
    end = input("Please input the desired running time of the simulation in seconds: ")
    step = input("Please input the desired time-step of the simulation in seconds: ")
    return float(end), float(step)


def split_list(target_list: list, offset: int, split: int) -> list:
    return_list = target_list.copy()
    if offset != 0:
        for i in range(offset):
            return_list.pop(0)
    return return_list[::split]
