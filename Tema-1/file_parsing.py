import re


def parse_equation():
    input_file_name = "input.txt"
    number_of_equations = 3
    number_of_variables = 3

    input_file = open(input_file_name, "r")
    string_equations = input_file.readlines()

    equations_parameters = []
    equations_result = []
    for i in range(0, number_of_equations):
        equations_parameters.append([])
        equations_result.append(0)
        for j in range(0, number_of_equations):
            equations_parameters[i].append(0)

    for i in range(0, number_of_equations):
        equations_result[i] = float(string_equations[i][string_equations[i].index("=") + 1:])
        string_equations[i] = string_equations[i][:string_equations[i].index("=")]

    coeficients = []
    for i in range(0, number_of_equations):
        regex_expression = re.compile("[+,-]?[0-9]+[xyz]+")
        coeficients.append(regex_expression.findall(string_equations[i]))

    for i in range(0, number_of_equations):
        for j in range(0, number_of_variables):
            coeficients[i][j] = int(coeficients[i][j][:-1])

    return coeficients, equations_result
