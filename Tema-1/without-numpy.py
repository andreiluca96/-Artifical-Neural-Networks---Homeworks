from file_parsing import parse_equation

number_of_equations = 3
number_of_variables = 3

def get_matrix(number_of_lines, number_of_columns):
    matrix = []
    for i in range(number_of_lines):
        matrix.append([])
        for j in range(number_of_columns):
            matrix[i].append(0)

    return matrix


def get_determinant(dimension, matrix):
    determinant = 0
    for i in range(dimension):
        left_diag = 1
        right_diag = 1
        for j in range(dimension):
            left_diag *= matrix[(i + j) % dimension][j]
            right_diag *= matrix[(i + j) % dimension][-(j + 1)]
        determinant += left_diag - right_diag
    return determinant


def get_transposed_matrix(number_of_lines, number_of_columns, matrix):
    transposed_matrix = get_matrix(number_of_lines, number_of_columns)
    for i in range(number_of_columns):
        for j in range(number_of_lines):
            transposed_matrix[i][j] = matrix[j][i]

    return transposed_matrix


def get_star_matrix_element(number_of_lines, number_of_columns, line, column, matrix):
    result_matrix = get_matrix(number_of_lines, number_of_columns)
    for i in range(number_of_lines):
        for j in range(number_of_columns):
            if i == line:
                break
            if j == column:
                continue
            result_matrix[i if i < line else i - 1][j if j < column else j - 1] = matrix[i][j]
    return (result_matrix[0][0] * result_matrix[1][1] - result_matrix[0][1] * result_matrix[1][0]) * (
        (-1) ** (line + column))


def get_matrix_multiplication(dimension1, dimension2, dimension3, matrix1, matrix2):
    result_matrix = get_matrix(dimension1, dimension3)
    for i in range(dimension1):
        for j in range(dimension3):
            for k in range(dimension2):
                result_matrix[i][j] += matrix1[i][k] * matrix2[j][k]
    return result_matrix


coeficients, equations_result = parse_equation()

determinant = get_determinant(number_of_equations, coeficients)

if determinant == 0:
    print "The system doesn't have solutions."
    exit(0)

transposed_coeficients = get_transposed_matrix(number_of_equations, number_of_variables, coeficients)

star_coeficients = get_matrix(number_of_equations, number_of_variables)

for i in range(number_of_equations):
    for j in range(number_of_variables):
        star_coeficients[i][j] = get_star_matrix_element(number_of_equations, number_of_variables, i, j,
                                                         transposed_coeficients)

equations_result_matrix = [equations_result]
result = get_matrix_multiplication(number_of_equations, number_of_variables, 1, star_coeficients,
                                   equations_result_matrix)
result = map(lambda x: x[0] / determinant, result)

print result
