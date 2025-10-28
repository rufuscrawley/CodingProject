import numpy as np
import matplotlib.pyplot as plt


def fc_operation(f, c):
    """
    Applies operation to find vector f as outlined in Gram-Schmidt process.
    :param f: Input vector
    :param c: Relevant column of matrix as a vector
    :return: Orthogonal f_vector for QU algorithm
    """
    f, c = np.array(f), np.array(c)
    dot_product = np.dot(f, c)
    normalisation_coefficient = np.power(np.linalg.norm(f), 2)
    # See W2:L2 lecture; slide 6 for equation
    return (dot_product / normalisation_coefficient) * f


def qu_algorithm(matrix, iterations):
    """
    Applies QU factorisation to find eigenvalues of a matrix.
    :param matrix: Matrix to act upon
    :param iterations: Number of iterations to perform algorithm over.
    :return:
    """
    output = matrix
    for iteration in range(iterations):
        # Transpose matrix into column vectors
        M_1 = np.transpose(output)

        # Construct a matrix of the f vectors.
        # Get current column, find normalisation coefficient for column,
        # iterate over remaining columns
        M_f = []
        for k, c_k in enumerate(M_1):
            # f_1 should always be the first column, so simply skip it
            if k == 0:
                M_f.append(c_k)
            else:
                # Calculate f_s - see lecture 4, slide 9, step 1
                f_k = np.array(M_1[k])
                for col_target in range(k):
                    if col_target > 0:
                        f_vector = fc_operation(M_f[k - 1], c_k)
                    else:
                        f_vector = M_1[col_target]
                    f_k -= fc_operation(f_vector, c_k)
                M_f.append(f_k)

        # Create our Q matrix from the F matrix
        # We could have directly made M_q in the above loops,
        # but storing the F matrix is useful for the next set of steps.
        M_q = []
        for element in M_f:
            M_q.append(element / np.linalg.norm(element))
        M_q = np.transpose(M_q)

        # Iterate over all columns in our Q matrix, preparing to set up U matrix
        M_u = []
        for M_q_x in range(len(M_q)):
            row_list = []
            # Then iterate over all rows in our Q matrix
            # Apply relevant operation depending on whether element is
            # above or below the leading diagonal
            for M_q_y in range(len(M_q)):
                if M_q_x > M_q_y:
                    # Above leading diagonal
                    row_list.append(0)
                if M_q_x == M_q_y:
                    # Leading diagonal
                    row_list.append(float(np.linalg.norm(np.array(M_f)[M_q_x])))
                if M_q_y > M_q_x:
                    # Below leading diagonal
                    value_to_add = np.dot(M_1[M_q_y], M_q[M_q_x])
                    row_list.append(float(value_to_add))
            M_u.append(row_list)

        output = np.matmul(M_u, M_q)
    # We have performed all necessary operations on a 2x2 matrix
    return output


def eigenvalues_to_frequency(masses: (int, int), spring_constant: int, index: int, iterations: int) -> int:
    """
    Performs the QU algorithm to find eigenvalues of a coupled harmonic oscillator.
    :param masses: Masses to use in our dual-mass setup, in kg.
    :param spring_constant: Second mass to use in our dual-mass setup, in kg.
    :param index: Indexes of the eigenvalues, inclusive between 1-0.
    :param iterations: Number of iterations to perform algorithm over.
    :return: The oscillation frequency of the masses, in Hz.
    """
    eigenvalue = qu_algorithm(
        [[-2 * spring_constant / masses[0], spring_constant / masses[1]],
         [spring_constant / masses[0], -2 * spring_constant / masses[1]]],
        iterations)[index][index]
    # Per the nature of our problem, we need to convert eigenvalues into frequencies.
    # Just in case - if a negative eigenvalue is returned, e.g. if not enough iterations are used,
    # sanitise it and return 0.
    if -eigenvalue > 0:
        return np.sqrt(-eigenvalue)
    else:
        return 0


# Graphing function
frequencies_1, frequencies_2 = [], []
correct_frequencies_1, correct_frequencies_2 = [], []
spaced_masses = np.arange(0.1, 20.0, 0.2)
# Graph against all applicable masses
for i in spaced_masses:
    frequencies_1.append(eigenvalues_to_frequency((i, i), 5, 0, 10000))
    frequencies_2.append(eigenvalues_to_frequency((i, i), 5, 1, 10000))
    correct_frequencies_1.append(np.sqrt(5 / i))
    correct_frequencies_2.append(np.sqrt(5 * 0.5 / i))

# Plot all masses
fig, ax = plt.subplots()
ax.plot(spaced_masses, frequencies_1, label='f_1')
ax.plot(spaced_masses, correct_frequencies_1, label='numeric f_1')
ax.plot(spaced_masses, frequencies_2, label='f_2')
ax.plot(spaced_masses, correct_frequencies_2, label='numeric f_2')

# Legends
ax.set(ylabel='frequency (Hz)', xlabel='mass (kg)',
       title='Coupled oscillator frequency')
ax.grid()
ax.legend()
plt.show()
