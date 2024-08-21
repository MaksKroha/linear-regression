import numpy as np
import matplotlib.pyplot as pyplot
from fileReader import *
from fileWriter import *


def derivative(value):
    return value * (1 - value)


def activate(mat):
    return 1 / (1 + np.exp(-mat))


def forwardCycle(input_vals, hidden_params, output_params):
    hidden_summa = np.dot(input_vals, hidden_params[0].T) + hidden_params[1]
    hidden_activated = activate(hidden_summa)

    output_summa = np.dot(hidden_activated, output_params[0].T) + output_params[1]
    output_activated = activate(output_summa)

    return hidden_activated, output_activated


def backpropagation(hidden_params, output_params, all_data, len_all_data, step, cycles):
    for _ in range(cycles):
        input_data = all_data[np.random.randint(0, 5000)]
        input_vals = np.array([input_data[0]])
        # Forward
        hidden_activated, output_activated = forwardCycle(input_vals, hidden_params, output_params)

        # Back
        d_error = output_activated - input_data[1]
        d_output_delta = d_error * derivative(output_activated)

        output_weights_gradients = d_output_delta * hidden_activated
        output_bias_gradients = d_output_delta

        d_hidden_delta = output_params[0] * d_output_delta * derivative(hidden_activated)
        hidden_weights_gradients = np.dot(d_hidden_delta.T, input_vals)
        hidden_bias_gradients = d_hidden_delta

        output_params[0] -= step * output_weights_gradients
        output_params[1] -= step * output_bias_gradients
        hidden_params[0] -= step * hidden_weights_gradients
        hidden_params[1] -= step * hidden_bias_gradients

    return [hidden_params, output_params]