import numpy as np
from tipo.Functional import activation, loss

class LinearPass:
    def __init__(self, num_inputs, num_neurons):
        self.weights = 0.10 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))

    def passData(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


