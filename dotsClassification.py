import numpy
import matplotlib.pyplot as drawer
dots = 5
bias = 3

x = numpy.random.randint(20, size=dots)
y = 3 * x + numpy.array([numpy.random.randint(1, 10) for _ in range(dots)]) + bias
higher_dots = numpy.array([x, y])

x = numpy.random.randint(20, size=dots)
y = 3 * x - numpy.array([numpy.random.randint(1, 10) for _ in range(dots)]) + bias
lower_dots = numpy.array([x, y])

weights = numpy.array([-3, 1, -bias])
for i in range(dots):
    res = numpy.dot(weights, numpy.array([lower_dots[0][i], lower_dots[1][i], 1]))
    if res > 0:
        print("C1")
    else:
        print("C2")

drawer.scatter(higher_dots[0], higher_dots[1], 20, "green", label="Higher dots")
drawer.scatter(lower_dots[0], lower_dots[1], 30, "red", label="Lower dots")
drawer.grid(True)
drawer.plot([i for i in range(-10, 20)], [i*3 + bias for i in range(-10, 20)])
drawer.show()

