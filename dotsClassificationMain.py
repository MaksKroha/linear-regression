import time

import matplotlib.pyplot as pyplot
import numpy as np
from fileWriter import *
from fileReader import *
import OptimizationMethods.miniBatchGradientDescent as mini_batch
import OptimizationMethods.stochasticGradientDescent as stochastic
import OptimizationMethods.momentumGradientDescent as momentum
import OptimizationMethods.nesterovMomentumGD as moment_nesterov
import OptimizationMethods.adagradGradientDescent as adagrad
import OptimizationMethods.RMSpropGradientDescent as rms_prop
import OptimizationMethods.adamGradientDescent as adam


def forwardCycle(input_vals, realization):
    params = readParameters("weights.json")
    hidden_summa = np.dot(input_vals, params[0][0].T) + params[0][1]
    hidden_activated = realization.activate(hidden_summa)
    output_summa = np.dot(hidden_activated, params[1][0].T) + params[1][1]
    output_activated = realization.activate(output_summa)

    return output_activated


def practice(dots, realization):
    x1 = np.array([np.random.randint(-1200, 2000) / 1000 for _ in range(dots)])
    x2 = np.array([np.random.randint(-1200, 2000) / 1000 for _ in range(dots)])
    coord = np.array([x1, x2])
    line1 = [[0, 0.5], [0.5, 0]]
    line2 = [[0.5, 1], [1, 0.5]]
    for i in range(dots):
        output = forwardCycle(np.array([coord[0][i], coord[1][i]]), realization)
        if output > 0.5:
            pyplot.scatter(coord[0][i], coord[1][i], 20, "orange")
        else:
            pyplot.scatter(coord[0][i], coord[1][i], 20, "blue")
        print(f"output - {output}")

    pyplot.grid(True)
    pyplot.plot(line1[0], line1[1], color="red")
    pyplot.plot(line2[0], line2[1], color="red")
    pyplot.show()


if __name__ == "__main__":
    data = readDataset("dataset.json")
    len_data = len(data)
    writeRandom("weights.json", 30)
    params = readParameters("weights.json")
    start = time.time()
    params = adam.adam_learning(params[0], params[1], data, len_data, 0.001, 1000)
    end = time.time()
    writeParams("weights.json", params[0], params[1])
    print(f"seconds - {end - start}")
    practice(50, adam)