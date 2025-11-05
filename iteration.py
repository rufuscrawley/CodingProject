from main_file import *

matrix = [[-1, 0.5],
          [0.5, -1]]
eigenvalues = get_eigenvalues(matrix, 5)

print(f"Eigenvalues are {eigenvalues[0][0]} and {eigenvalues[1][1]}")
