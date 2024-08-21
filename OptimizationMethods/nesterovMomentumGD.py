import math
import numpy as np


def activate(mat):
    return 1 / (1 + np.exp(-mat))


def derivative(value):
    return value * (1 - value)


# Based on mini-batch gradient descent
def momentum_nesterov_learning(hidden_params, output_params, all_data, len_all_data, step, cycles):
    quality_const = 64
    hidden_speed = [np.zeros(hidden_params[0].shape), np.zeros(hidden_params[1].shape)]
    output_speed = [np.zeros(output_params[0].shape), np.zeros(output_params[1].shape)]
    momentum = 0.9
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
            future_hidden_params = [param.copy() for param in hidden_params]
            future_output_params = [param.copy() for param in output_params]
            future_hidden_params[0] -= momentum * hidden_speed[0]
            future_hidden_params[1] -= momentum * hidden_speed[1]
            future_output_params[0] -= momentum * output_speed[0]
            future_output_params[1] -= momentum * output_speed[1]

            input_expected_vals = [[], []]
            for data in mini_batch:
                input_expected_vals[0].append(data[0])
                input_expected_vals[1].append([data[1]])
            # Forward Propagation
            input_expected_vals[0] = np.array(input_expected_vals[0])
            input_expected_vals[1] = np.array(input_expected_vals[1])
            hidden_summa = np.dot(input_expected_vals[0], future_hidden_params[0].T) + future_hidden_params[1]
            hidden_activated = activate(hidden_summa)

            output_summa = np.dot(hidden_activated, future_output_params[0].T) + future_output_params[1]
            output_activated = activate(output_summa)

            # Back Propagation
            d_error = output_activated - input_expected_vals[1]
            new_error += np.mean(np.abs(d_error))
            d_output_delta = d_error * derivative(output_activated)
            output_speed[0] = 0.9 * output_speed[0] + step * np.dot(d_output_delta.T, hidden_activated) / quality_const
            output_speed[1] = 0.9 * output_speed[1] + step * np.mean(d_output_delta)
            output_params[0] -= 0.9 * output_speed[0]
            output_params[1] -= 0.9 * output_speed[1]

            d_hidden_delta = future_output_params[0] * d_output_delta * derivative(hidden_activated)
            hidden_speed[0] = 0.9 * hidden_speed[0] + step * np.dot(d_hidden_delta.T, input_expected_vals[0]) / quality_const
            hidden_speed[1] = 0.9 * hidden_speed[1] + step * np.mean(d_hidden_delta, axis=0)
            hidden_params[0] -= 0.9 * hidden_speed[0]
            hidden_params[1] -= 0.9 * hidden_speed[1]
    return hidden_params, output_params