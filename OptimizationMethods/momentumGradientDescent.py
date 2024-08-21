import math
import numpy as np


def activate(mat):
    return 1 / (1 + np.exp(-mat))


def derivative(value):
    return value * (1 - value)


# Based on mini-batch gradient descent
def momentum_learning(hidden_params, output_params, all_data, len_all_data, step, cycles):
    quality_const = 64
    hidden_speed = [np.zeros(hidden_params[0].shape), np.zeros(hidden_params[1].shape)]
    output_speed = [np.zeros(output_params[0].shape), np.zeros(output_params[1].shape)]
    old_error = 0
    new_error = 1
    for _ in range(cycles):
        # Check if should stop learning
        new_error /= math.ceil(len_all_data / quality_const)
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
            output_speed[0] = 0.9 * output_speed[0] + step * np.dot(d_output_delta.T, hidden_activated) / quality_const
            output_speed[1] = 0.9 * output_speed[1] + step * np.mean(d_output_delta)
            output_params[0] -= output_speed[0]
            output_params[1] -= output_speed[1]

            d_hidden_delta = output_params[0] * d_output_delta * derivative(hidden_activated)
            hidden_speed[0] = 0.9 * hidden_speed[0] + step * np.dot(d_hidden_delta.T, input_vals) / quality_const
            hidden_speed[1] = 0.9 * hidden_speed[1] + step * np.mean(d_hidden_delta, axis=0)
            hidden_params[0] -= hidden_speed[0]
            hidden_params[1] -= hidden_speed[1]
    return hidden_params, output_params


