import json as js
import numpy as np


def readParameters(fileName):
    with open(fileName) as file:
        all_content = js.load(file)
        return [[np.array(all_content['param1'][0]), np.array(all_content['param1'][1])],
                [np.array(all_content['param2'][0]), np.array(all_content['param2'][1])]]


def readDataset(fileName) -> [[]]:
    with open(fileName) as file:
        all_content = js.load(file)
        return all_content['dataset']
