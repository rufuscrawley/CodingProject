import numpy as np
import matplotlib.pyplot as plt

# region DEBUGGING

# debug flags
# use if you want to see various debug info
# recommended to not use with high iteration counts!
debug = False
debug_fc = False


# endregion

# region HELPER FUNCTIONS

def fc_operation(f, c):
    """
    Applies operation to find vector f as outlined in Gram-Schmidt process.
    :param f: Input vector
    :param c: Relevant column of matrix as a vector
    :return: Orthogonal f_vector for QU algorithm
    """
    f, c = np.array(f), np.array(c)
    dot_product = np.dot(f, c)
    normalisation_coefficient = np.power(np.linalg.norm(f), 2).round(5)
    # See W2:L2 lecture; slide 6 for equation
    if debug_fc:
        print("=== FC STUFF ===")
        print(f"The dot product of these vectors was {dot_product}")
        print(f"Normalising f gave me {normalisation_coefficient}")
        print(f"Performing: ({dot_product} / {normalisation_coefficient}) * {f}")
        print(f"Thus, I return {(dot_product / normalisation_coefficient) * f}")
        print("=== END FC STUFF ===")
    return (dot_product / normalisation_coefficient) * f


def qu_algorithm(matrix):
    """
    Applies QU factorisation to find eigenvalues of a matrix.
    :param matrix: Matrix to act upon
    """
    # Transpose matrix into column vectors

    if debug:
        print(f"Working on: {matrix}")
    matrix_t = np.transpose(matrix)

    # Construct a matrix of the f vectors.
    # Get current column, find normalisation coefficient for column,
    # iterate over remaining columns
    M_f = []
    for k, c_k in enumerate(matrix_t):
        # f_1 should always be the first column, so simply skip it
        if k == 0:
            M_f.append(c_k)
        else:
            # Calculate f_s - see lecture 4, slide 9, step 1
            f_k = c_k
            for col_target in range(k):
                if col_target > 0:
                    f_vector = fc_operation(M_f[k - 1], c_k)
                else:
                    f_vector = matrix_t[col_target]
                f_k -= fc_operation(f_vector, c_k)
            M_f.append(f_k)

    if debug:
        print(f"M_f: {M_f}")

    # Create our Q matrix from the F matrix
    # We could have directly made M_q in the above loops,
    # but storing the F matrix is useful for the next set of steps.
    M_q = []
    for col in M_f:
        if np.linalg.norm(col) == 0:
            M_q.append(col * np.linalg.norm(col))
        else:
            M_q.append(col / np.linalg.norm(col))
    M_q = np.transpose(M_q)

    if debug:
        print(f"M_q: {M_q}")

    # Iterate over all columns in our Q matrix, preparing to set up U matrix
    M_u = []
    for M_q_x in range(len(M_q)):
        row_list = []
        # Then iterate over all rows in our Q matrix
        # Apply relevant operation depending on whether element is
        # above or below the leading diagonal
        for M_q_y in range(len(M_q)):
            if M_q_x > M_q_y:
                # Below leading diagonal
                row_list.append(0.0)
            if M_q_x == M_q_y:
                # On the leading diagonal
                row_list.append(np.linalg.norm(np.array(M_f)[M_q_x]).round(5))
            if M_q_y > M_q_x:
                # Above leading diagonal
                value_to_add = np.dot(np.transpose(matrix)[M_q_y], np.transpose(M_q)[M_q_x]).round(5)
                row_list.append(value_to_add)
        M_u.append(row_list)
    # We have performed all necessary operations on a 2x2 matrix
    output = M_u @ M_q
    if debug:
        print(f"M_u: {M_u}")
        print(f"M_q: {M_q}")
        print(f"output: {output}\n============================")
    return output


def get_eigenvalues(target_matrix, iterations: int):
    """
    Performs the QU algorithm to find eigenvalues of a coupled harmonic oscillator.
    :param target_matrix: The matrix to get the eigenvalues of
    :param iterations: Number of iterations to perform algorithm over.
    :return: 2x2 matrix with the eigenvalues on the leading diagonal
    """
    # Note: I extracted the logic from qu_algorithm() because I felt the control flow of the code was
    # slightly messier than I would prefer - all this function does is wrap a for-loop.
    matrix = target_matrix
    for iteration in range(iterations):
        matrix = qu_algorithm(
            [[matrix[0][0], matrix[0][1]],
             [matrix[1][0], matrix[1][1]]])
    return matrix


def harmonic_matrix(m_1, m_2, k):
    """
    Helper function for formatting a matrix for the assignment/homework. Simply spits out a matrix if given
    the physical parameters.
    :param m_1: First mass
    :param m_2: Second mass
    :param k: Spring constant
    """
    # See PHYM004_Assessment1.pdf for the matrix equation
    return [[(-2 * k / m_1), (k / m_2)],
            [(k / m_1), (-2 * k / m_2)]]


def graph_eigenvalues(spring_constant, iteration_count):
    """
    Helper function to handle everything graph-related in the main body. Mainly used for testing.
    :param spring_constant: ...Spring constant
    :param iteration_count: Number of iterations to perform the QU algorithm over.
    """
    # Set up empty lists
    frequencies_1, frequencies_2 = [], []
    correct_frequencies_1, correct_frequencies_2 = [], []
    # Set up masses for the graph - could make configurable but beyond project scope
    spaced_masses = np.arange(0.1, 20.0, 0.2)
    # Graph against all applicable masses
    for i in spaced_masses:
        # Graph my code's calculated eigenvalue frequencies for the problem
        lambda_1, lambda_2 = (get_eigenvalues(harmonic_matrix(i, i, spring_constant), iteration_count)[0][0],
                              get_eigenvalues(harmonic_matrix(i, i, spring_constant), iteration_count)[1][1])
        frequencies_1.append(np.sqrt(-lambda_1))
        frequencies_2.append(np.sqrt(-lambda_2))

        # Graph "correct" frequencies against analytical solution for the problem
        correct_frequencies_1.append(np.sqrt(spring_constant / i))
        correct_frequencies_2.append(np.sqrt(spring_constant * 3 / i))

    # Plot all masses
    fig, ax = plt.subplots()
    ax.plot(spaced_masses, frequencies_1, label='f_1')
    # ax.plot(spaced_masses, correct_frequencies_1, label='numeric f_1')
    ax.plot(spaced_masses, frequencies_2, label='f_2')
    # ax.plot(spaced_masses, correct_frequencies_2, label='numeric f_2')

    fig_2, ax_2 = plt.subplots()
    differences = []
    for number, frequency in enumerate(frequencies_1):
        differences.append(frequency - correct_frequencies_2[number])
    differences_2 = []
    for number, frequency in enumerate(frequencies_2):
        differences_2.append(frequency - correct_frequencies_1[number])
    ax_2.plot(spaced_masses, differences_2)
    ax_2.set(ylabel='deviation (Hz)', xlabel='mass (kg)',
             title='Differences')
    ax_2.legend()
    ax_2.grid()

    # Legends
    ax.set(ylabel='frequency (Hz)', xlabel='mass (kg)',
           title='Coupled oscillator frequency')
    ax.grid()
    ax.legend()
    plt.show()


def input_sanitised(request_string, desired_type):
    """
    Lazy function to sanitise user input (once). Could be wrapped in a loop instead but probably not in project scope
    :param request_string: The request made to the user [for input].
    :param desired_type: The return type desired. If non-castable, an exception will be caught.
    :return:
    """
    try:
        input_string = input(request_string)
        return desired_type(input_string)
    except TypeError:
        print("Error!")
        input_string = input(request_string)


# endregion


# region MAIN BODY OF CODE

print("Graphing mode graphs two equal masses with a configurable spring constant.")
print("Alternately you can input manual masses/spring constant.")
graphing_mode = input_sanitised("Would you like graphing mode? (y/n)\n", str)
if graphing_mode == "y":
    graphing_mode = True
elif graphing_mode == "n":
    graphing_mode = False

if graphing_mode:
    print("=== Graphing mode selected! ===")
    # Prints a graph of m against frequency, alongside a graph for the deviation for numerical vs. QU
    k = input_sanitised("Input spring constant: ", float)
    it = input_sanitised("Input number of iterations (more=slower but more accurate): ", int)
    graph_eigenvalues(k, it)
elif not graphing_mode:
    print("=== Manual mode selected ===")
    # Mostly for debugging, but allows you to get manual eigenvalues for this problem
    mass_1 = input_sanitised("Input mass 1: ", float)
    mass_2 = input_sanitised("Input mass 2: ", float)
    spring_constant = input_sanitised("Input spring constant: ", float)
    it = input_sanitised("Input number of iterations (more=slower but more accurate): ", int)
    print(f"Working with m_1 = {mass_1} kg, m_2 = {mass_2} kg, k = {spring_constant} Nm-1")
    eigenvalue_1 = get_eigenvalues(harmonic_matrix(mass_1, mass_2, spring_constant), it)[0][0]
    print(f"First eigenvalue: {np.sqrt(-eigenvalue_1).round(5)} Hz")
    eigenvalue_2 = get_eigenvalues(harmonic_matrix(mass_1, mass_2, spring_constant), it)[1][1]
    print(f"Second eigenvalue: {np.sqrt(-eigenvalue_2).round(5)} Hz")
    if mass_1 == mass_2:
        print(f"Classically calculated eigenvalues: "
              f"{np.sqrt(spring_constant / mass_1).round(5)} Hz, "
              f"{np.sqrt((spring_constant * 3) / mass_2).round(5)} Hz")

# endregion
