import math
import time
import numpy as np
from fileReader import *
from fileWriter import *


def activate(mat):
    return 1 / (1 + np.exp(-mat))


def derivative(activated):
    return activated * (1 - activated)


def miniBatch(hidden_params, output_params, all_data, len_all_data, step, cycles):
    quality_const = 64
    old_error = 0
    new_error = 1
    for _ in range(cycles):
        new_error /= math.ceil(len_all_data / quality_const)
        # Check if should stop learning
        if math.floor(old_error * 1000000) == math.floor(new_error * 1000000):
            break
        old_error, new_error = new_error, 0
        np.random.shuffle(all_data)
        mini_batches = []
        for i in range(0, len_all_data, quality_const):
            mini_batches.append(all_data[i: i + quality_const])

        for mini_batch in mini_batches:
            input_vals = []
            expected_vals = []
            for data in mini_batch:
                input_vals.append(data[0])
                expected_vals.append([data[1]])
            # Forward Propagation
            input_vals = np.array(input_vals)
            expected_vals = np.array(expected_vals)
            hidden_summa = np.dot(input_vals, hidden_params[0].T) + hidden_params[1]
            hidden_activated = activate(hidden_summa)

            output_summa = np.dot(hidden_activated, output_params[0].T) + output_params[1]
            output_activated = activate(output_summa)

            # Back Propagation
            d_error = output_activated - expected_vals
            new_error += np.mean(np.abs(d_error))
            d_output_delta = d_error * derivative(output_activated)

            output_params[0] -= step * np.dot(d_output_delta.T, hidden_activated) / quality_const
            output_params[1] -= step * np.mean(d_output_delta)

            d_hidden_delta = output_params[0] * d_output_delta * derivative(hidden_activated)
            hidden_params[0] -= step * np.dot(d_hidden_delta.T, input_vals) / quality_const
            hidden_params[1] -= step * np.mean(d_hidden_delta, axis=0)
    return hidden_params, output_params


if __name__ == "__main__":
    params = readParameters("../weights.json")
    dataset = readDataset("../dataset.json")
    hidden_params, output_params = miniBatch(params[0], params[1], dataset, len(dataset), 0.1, 1000)
    writeParams("../weights.json", hidden_params, output_params)