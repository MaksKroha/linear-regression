import numpy as np
import json as js


def writeRandom(fileName, neurons):
    #                           (вихід, вхід)
    hidden_weights = np.random.uniform(-0.4, 0.4, size=(neurons, 2))
    hidden_bias = np.random.uniform(-1, 1, size=(1, neurons))
    output_weights = np.random.uniform(-0.4, 0.4, size=(1, neurons))
    output_bias = np.random.uniform(-1, 1, size=(1, 1))
    to_json = {"param1": [hidden_weights.tolist(), hidden_bias.tolist()],
               "param2": [output_weights.tolist(), output_bias.tolist()]}
    with open(fileName, 'w') as file:
        js.dump(to_json, file)


def writeParams(fileName, hidden_params, output_params):
    to_json = {"param1": [hidden_params[0].tolist(), hidden_params[1].tolist()],
               "param2": [output_params[0].tolist(), output_params[1].tolist()]}
    with open(fileName, 'w') as file:
        js.dump(to_json, file)


def writeDataset(fileName, dots):
    data_set = []

    x1 = np.array([np.random.randint(-2000, 2000) / 1000 for _ in range(dots)])
    x2 = np.array([np.random.randint(-2000, 2000) / 1000 for _ in range(dots)])
    bias = 1
    coord = np.array([x1, x2])

    weights1 = np.array([[1, 1, -3 / 2], [1, 1, -1 / 2]])
    weights2 = np.array([-1, 1])
    line1 = [[0, 0.5], [0.5, 0]]
    line2 = [[0.5, 1], [1, 0.5]]

    def activate(sum):
        return 0 if sum <= 0 else 1

    for i in range(dots):
        hidden_res = np.dot(weights1, [coord[0][i], coord[1][i], bias])
        hidden_res = np.array([activate(x) for x in hidden_res])
        res = np.dot(weights2, hidden_res)
        if activate(res) == 1:
            data_set.append([[coord[0][i], coord[1][i]], 1])
            # pyplot.scatter(coord[0][i], coord[1][i], 20, "orange")
        else:
            data_set.append([[coord[0][i], coord[1][i]], 0])
            # pyplot.scatter(coord[0][i], coord[1][i], 20, "blue")

    with open(fileName, 'w') as file:
        js.dump({'dataset': data_set}, file)

    # pyplot.grid(True)
    # pyplot.plot(line1[0], line1[1], color="red")
    # pyplot.plot(line2[0], line2[1], color="red")
    # pyplot.show()