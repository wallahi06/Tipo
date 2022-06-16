import numpy as np


class Activation:

    def __init__(self):
        pass

    def binarystep(value):
        return np.heaviside(value, 1)

    def linear(self, value, gradient=4):
        return gradient*value

    def sigmoid(self, value):
        return 1/(1 + np.exp(-value))

    def tanh(self, value):
        return 2/(1 + np.exp(-2*value)) - 1

    def relu(self, value, gradient=1):
        if value < 0:
            return 0
        else:
            return value*gradient

    def leakyrelu(self, value, gradient=1, leak=0.01):
        if value < 0:
            return value*leak
        else:
            return value*gradient

    def elu(self, value, a=0.1):
        if value < 0:
            return a*(np.exp(value)-1)
        else:
            return value

    def swish(self, value):
        return value/(1-np.exp(-value))

    def softmax(self, value):
        z = np.exp(value)
        value_ = z / z.sum()
        return value_

activation = Activation()

class Loss:

    def __init__(self):
        pass

    def meanAbsoluteError(self, true_value, prediction):
        true_value, prediction = np.array(true_value), np.array(prediction)
        return np.mean(np.abs(true_value - prediction))

    def meanSquareError(self, true_value, prediction):
        true_value, prediction = np.array(true_value), np.array(prediction)
        return np.square(true_value - prediction).mean()

    
loss = Loss()
