import cPickle, gzip, numpy

input_size = 784

number_of_digits = 10
nr_iterations = 10
learning_rate = 0.1

bias = numpy.random.rand(input_size)
weights = numpy.random.normal(0.5, 1, (number_of_digits, input_size))

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()


for iteration in range(0, nr_iterations):
    for i in range(len(train_set[0])):
        print str(iteration) + ' ' + str(i)
        for digit in range(0, number_of_digits):
            z = numpy.dot(train_set[0][i], weights[digit])
            z += bias[digit]

            net_output = 0
            if z > 0:
                net_output = 1

            weights[digit] = numpy.add(weights[digit],
                                       (int(train_set[1][i] == digit) - net_output) * learning_rate * train_set[0][i])
            bias[digit] += (int(train_set[1][i] == digit) - net_output) * learning_rate

successes = 0
fails = 0

for i in range(0, len(test_set[0])):
    print i
    results = [0 for j in range(0, number_of_digits)]
    for digit in range(0, number_of_digits):
        results[digit] = numpy.dot(test_set[0][i], weights[digit])
        results[digit] += bias[digit]

    if test_set[1][i] == numpy.argmax(results):
        successes += 1
    else:
        fails += 1

print "Successes: " + str(successes)
print "Fails: " + str(fails)
print float(successes) / float((successes + fails)) * 100
