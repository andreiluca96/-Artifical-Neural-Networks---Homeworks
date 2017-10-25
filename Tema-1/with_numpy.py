import numpy as np

from file_parsing import parse_equation

number_of_equations = 3
number_of_variables = 3

coeficients, equations_result = parse_equation()

# Method 1
print np.linalg.solve(coeficients, equations_result)

# Method 2
determinant = np.linalg.det(coeficients)

if determinant == 0:
    print "The system doesn't have solutions."
    exit(0)

matrix_inverse = np.linalg.inv(coeficients)

result = np.dot(matrix_inverse, equations_result)
print result