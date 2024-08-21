import math
import time
import numpy as np
from fileReader import *
from fileWriter import *


def activate(mat):
    return 1 / (1 + np.exp(-mat))


def derivative(activated):
    return activated * (1 - activated)


def adam_learning(hidden_params, output_params, all_data, len_all_data, step, cycles):
    quality_const = 64
    average = [[np.zeros_like(hidden_params[0]), np.zeros_like(hidden_params[1])],
               [np.zeros_like(output_params[0]), np.zeros_like(output_params[1])]]
    average2 = [[np.zeros_like(hidden_params[0]), np.zeros_like(hidden_params[1])],
                [np.zeros_like(output_params[0]), np.zeros_like(output_params[1])]]
    average_b = 0.9
    average2_b = 0.999
    clip = 10000
    je = 10 ** -7
    t = 0
    old_error = 0
    new_error = 1

    for _ in range(cycles):
        # Check if it should stop learning
        new_error /= math.ceil(len_all_data / quality_const)
        if math.floor(old_error * 700000) == math.floor(new_error * 700000):
            print("-------------------")
            print(f"{average[1][1]}     {average2[1][1]}")
            print("-------------------")
            break
        old_error, new_error = new_error, 0

        np.random.shuffle(all_data)
        mini_batches = []
        for i in range(0, len_all_data, quality_const):
            mini_batches.append(all_data[i: i + quality_const])

        for mini_batch in mini_batches:
            t += 1
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
            gradients = [np.dot(d_output_delta.T, hidden_activated) / quality_const,
                         np.mean(d_output_delta)]
            for i in range(2):
                average[1][i] = (average_b * average[1][i] + (1 - average_b) * gradients[i]) / (1 - average_b ** t)
                average2[1][i] = (average2_b * average2[1][i] + (1 - average2_b) * gradients[i] ** 2) / (1 - average2_b ** t)
                average[1][i] = np.clip(average[1][i], -clip, clip)
                average2[1][i] = np.clip(average2[1][i], -clip, clip)
                output_params[i] -= step * average[1][i] / (np.sqrt(average2[1][i]) + je)

            d_hidden_delta = output_params[0] * d_output_delta * derivative(hidden_activated)
            gradients = [np.dot(d_hidden_delta.T, input_vals) / quality_const,
                         np.mean(d_hidden_delta, axis=0)]

            for i in range(2):
                average[0][i] = (average_b * average[0][i] + (1 - average_b) * gradients[i]) / (1 - average_b ** t)
                average2[0][i] = (average2_b * average2[0][i] + (1 - average2_b) * gradients[i] ** 2) / (1 - average2_b ** t)
                average[0][i] = np.clip(average[0][i], -clip, clip)
                average2[0][i] = np.clip(average2[0][i], -clip, clip)
                # print(f"{average[1][i]}    {average2[1][i]}")

                hidden_params[i] -= step * average[0][i] / (np.sqrt(average2[0][i]) + je)

    return hidden_params, output_params
