import cPickle, gzip, numpy, math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


array_sigmoid = numpy.vectorize(sigmoid)

input_size = 784

number_of_digits = 10
nr_iterations = 10
hidden_layer_size = 10
learning_rate = 0.1
number_of_layers = 2

layer_weights = [numpy.random.normal(1 / 2, 1, (input_size, hidden_layer_size)),
                 numpy.random.normal(1 / 2, 1, (hidden_layer_size, number_of_digits))]
biases = [numpy.zeros(hidden_layer_size), numpy.zeros(number_of_digits)]

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

for iteration in range(nr_iterations):
    for train_number in range(len(train_set[0])):
        print str(iteration) + " " + str(train_number)

        layer_results = [numpy.zeros(hidden_layer_size), numpy.zeros(number_of_digits)]
        layer_results[0] = array_sigmoid(numpy.dot(train_set[0][train_number], layer_weights[0]) + biases[0])

        aux = numpy.exp(numpy.dot(layer_results[0], layer_weights[1]) + biases[1])
        layer_results[1] = aux / sum(aux)





