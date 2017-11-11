import cPickle, gzip, numpy, math


def cost_derivative(output_activations, y):
    return output_activations - y


def sigmoid(z):
    return 1.0 / (1.0 + numpy.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


array_sigmoid = numpy.vectorize(sigmoid)

input_size = 784

number_of_digits = 10
nr_iterations = 1
hidden_layer_size = 100
learning_rate = 0.1
number_of_layers = 2
regularization_rate = 0.1
friction_rate = 0.1

layer_weights = [numpy.random.normal(0, 1 / (math.sqrt(input_size)), (input_size, hidden_layer_size)),
                 numpy.random.normal(0, 1 / (math.sqrt(hidden_layer_size)), (hidden_layer_size, number_of_digits))]

layer_weights_speed = [numpy.zeros((input_size, hidden_layer_size)), numpy.zeros((hidden_layer_size, number_of_digits))]

biases = [numpy.zeros(hidden_layer_size), numpy.zeros(number_of_digits)]

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

for iteration in range(nr_iterations):
    for train_number in range(len(train_set[0])):
        train_input = []
        for i in train_set[0][train_number]:
            train_input.append(i)

        train_input = numpy.reshape(train_input, (784, 1))

        print str(iteration) + " " + str(train_number)

        activation_layer_results = [numpy.zeros(hidden_layer_size), numpy.zeros(number_of_digits)]
        layer_results = [numpy.zeros(hidden_layer_size), numpy.zeros(number_of_digits)]
        layer_errors = [numpy.zeros(hidden_layer_size), numpy.zeros(number_of_digits)]

        layer_results[0] = numpy.dot(train_set[0][train_number], layer_weights[0]) + biases[0]
        activation_layer_results[0] = numpy.array(map(lambda x: sigmoid(x), layer_results[0]))

        layer_results[1] = numpy.dot(activation_layer_results[0], layer_weights[1]) + biases[1]
        activation_layer_results[1] = numpy.array(map(lambda x: sigmoid(x), layer_results[1]))

        # softmax last layer error
        layer_errors[1] = numpy.exp(layer_results[1]) / numpy.sum(numpy.exp(layer_results[1]))
        layer_errors[1][train_set[1][train_number]] -= 1

        # classic sigmoid error
        # for digit in range(0, number_of_digits):
        #     layer_errors[1][digit] = (activation_layer_results[1][digit] - float(int(train_set[1][train_number] == digit))) \
        #                                                 * (1 - activation_layer_results[1][digit]) * activation_layer_results[1][digit]

        layer_errors[0] = (1 - activation_layer_results[0]) * activation_layer_results[0] \
                          * numpy.dot(layer_errors[1], layer_weights[1].transpose())

        layer_errors_reshaped_1 = numpy.reshape(layer_errors[0], (hidden_layer_size, 1))
        layer_errors_reshaped_2 = numpy.reshape(layer_errors[1], (number_of_digits, 1))
        activation_layer_results_reshaped_1 = numpy.reshape(activation_layer_results[0], (hidden_layer_size, 1))

        # momentum
        layer_weights_speed[0] = layer_weights_speed[0] * friction_rate - \
                                 numpy.dot(train_input, numpy.transpose(layer_errors_reshaped_1)) * learning_rate
        layer_weights_speed[1] = layer_weights_speed[1] * friction_rate -\
                                 numpy.dot(activation_layer_results_reshaped_1, numpy.transpose(layer_errors_reshaped_2)) * learning_rate

        # L2 regularization
        layer_weights[0] *= (1 - learning_rate * (regularization_rate / len(layer_weights[0])))
        layer_weights[1] *= (1 - learning_rate * (regularization_rate / len(layer_weights[1])))

        layer_weights[0] += layer_weights_speed[0]
        layer_weights[1] += layer_weights_speed[1]

        biases[0] -= layer_errors[0] * learning_rate
        biases[1] -= layer_errors[1] * learning_rate

successes = 0
fails = 0
for i in range(0, len(test_set[0])):
    activation_layer_results = [numpy.zeros(hidden_layer_size), numpy.zeros(number_of_digits)]
    layer_results = [numpy.zeros(hidden_layer_size), numpy.zeros(number_of_digits)]

    layer_results[0] = numpy.dot(test_set[0][i], layer_weights[0]) + biases[0]
    activation_layer_results[0] = numpy.array(map(lambda x: sigmoid(x), layer_results[0]))
    layer_results[1] = numpy.dot(activation_layer_results[0], layer_weights[1]) + biases[1]
    activation_layer_results[1] = numpy.array(map(lambda x: sigmoid(x), layer_results[1]))

    print numpy.argmax(activation_layer_results[1])

    if test_set[1][i] == numpy.argmax(activation_layer_results[1]):
        successes += 1
    else:
        fails += 1

print "Successes: " + str(successes)
print "Fails: " + str(fails)
print str(float(successes) / float((successes + fails)) * 100) + "%"
