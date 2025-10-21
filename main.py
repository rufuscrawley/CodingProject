import numpy as np
import matplotlib.pyplot as plt


def f_c_stuff(f, c):
    f, c = np.array(f), np.array(c)
    dot_product = np.dot(f, c)
    normalisation_coefficient = (np.pow(np.linalg.vector_norm(f), 2))
    return (dot_product / normalisation_coefficient) * f


# [1, 1]
# [2, 0]
M_1 = [[1, 1], [2, 0]]


def qu_algorithm(matrix, iterations):
    output = matrix
    for i in range(iterations):
        # Transpose matrix into column vectors
        M_1 = np.transpose(output)

        # Get current column, find normalisation coefficient for column, iterate over remaining columns
        M_f = []
        iteration = 0
        for column_number, column in enumerate(M_1):
            new_column = []
            if column_number > 0:
                # Start by finding c_k.
                c_k = column
                vector_result = np.array(M_1[column_number])
                for i in range(column_number):
                    if i > 0:
                        f_vector = f_c_stuff(M_f[column_number - 1], c_k)
                    else:
                        f_vector = M_1[i]
                    subtractor = f_c_stuff(f_vector, c_k)

                    vector_result = vector_result - subtractor
                M_f.append(vector_result)
            else:
                M_f.append(column)

        # Excellent!

        M_q = []
        for element in M_f:
            M_q.append(element / np.linalg.vector_norm(element))

        M_q = np.transpose(M_q)
        M_u = []

        for i in range(len(M_q)):
            row_list = []
            for j in range(len(M_q)):

                if j < i:
                    row_list.append(0)
                if i == j:
                    row_list.append(float(np.linalg.norm(np.array(M_f)[i])))
                if i < j:
                    value_to_add = np.dot(M_1[j], M_q[i])
                    row_list.append(float(value_to_add))
            M_u.append(row_list)

        output = np.linalg.matmul(M_u, M_q)
    return output


masses = np.arange(0.1, 20.0, 0.2)
spring_constant = 5


def eigenvalues_to_frequency(mass, index, iterations):
    eigenvalue = qu_algorithm(
        [[-2 * spring_constant / mass, spring_constant / mass],
         [spring_constant / mass, -2 * spring_constant / mass]],
        iterations)[index][index]
    if -eigenvalue > 0:
        return np.sqrt(-eigenvalue)
    else:
        return 0


frequencies_1, frequencies_2 = [], []
correct_frequencies_1, correct_frequencies_2 = [], []
for i in masses:
    frequencies_1.append(eigenvalues_to_frequency(i, 0, 1000))
    frequencies_2.append(eigenvalues_to_frequency(i, 1, 1000))
    correct_frequencies_1.append(np.sqrt(spring_constant / i))
    correct_frequencies_2.append(np.sqrt(spring_constant * 0.5 / i))

fig, ax = plt.subplots()
ax.plot(masses, frequencies_1, label='f_1')
ax.plot(masses, correct_frequencies_1, label='numeric f_1')
ax.plot(masses, frequencies_2, label='f_2')
ax.plot(masses, correct_frequencies_2, label='numeric f_2')

ax.set(ylabel='frequency (Hz)', xlabel='mass (kg)',
       title='Coupled oscillator frequency')
ax.grid()
ax.legend()
plt.show()
