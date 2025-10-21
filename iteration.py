import numpy as np


def f_c_stuff(f_vector, c_vector):
    f_array, c_array = np.array(f_vector), np.array(c_vector)
    dot_product = np.dot(f_array, c_array)
    normalisation_coefficient = (np.pow(np.linalg.vector_norm(f_array), 2))
    return (dot_product / normalisation_coefficient) * f_array


print(f_c_stuff([0.5, 0.5, 1, 0], [0, 1, 1, 1]))
