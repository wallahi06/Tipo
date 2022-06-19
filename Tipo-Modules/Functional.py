import numpy as np


# This is the base class for all the loss and cost functions
class Loss:
    def __init__(self, true_value, prediction):
        self.true_value = np.array(true_value)
        self.prediction = np.array(prediction)

    def __call__(self):
        pass

    def backward(self):
        pass


# This is a child class of the loss class, it calculates the MSE Loss
class MSELoss(Loss):
    def __init__(self, true_value, prediction):
        super().__init__(true_value, prediction)

    def __call__(self):
        return np.mean(np.power(self.true_value - self.prediction, 2))

    def backward(self):
        return 2 * (self.prediction - self.true_value) / np.size(self.true_value)


# This is a child class of the loss class, it calculates the MAE Loss
class MAELoss(Loss):
    def __init__(self, true_value, prediction):
        super().__init__(true_value, prediction)

    def __call__(self):
        return np.mean(np.abs(self.true_value - self.prediction))

    def backward(self):
        pass


# This is a child class of the loss class, it calculates the MBE Loss
class MBELoss(Loss):
    def __init__(self, true_value, prediction):
        super().__init__(true_value, prediction)

    def __call__(self):
        return (self.true_value - self.prediction).mean()

    def backward(self):
        pass


# This is the main class for all the activation functions
class Activation:

    # We declare the input, activation function and the prime activation function
    def __init__(self, input, activation=None, d_activation=None):
        self.input = np.array(input)
        self.activation = activation
        self.d_activation = d_activation

    # Passes data through with the selected activation
    def passdata(self):
        return self.activation

    # Passes data through with the selected prime activation
    def backward(self):
        return self.d_activation


# # # # # # # # # # # # # # # #
# Linear Activation functions #
# # # # # # # # # # # # # # # # 


# This is a child class of the Activation class that applies the ReLu Activation to its input
class ReLu(Activation):

    def __init__(self, input, gradient=1):
        super().__init__(input)
        self.gradient = gradient
        activation = np.where(self.input > 0, self.input * self.gradient, 0)
        d_activation = (self.input > 0) * 1
        super().__init__(input, activation, d_activation)


# This is a child class of the Activation class that applies the ReLu Activation to its input
class Sigmoid(Activation):

    def __init__(self, input):
        super().__init__(input)
        activation = 1 / (1 + np.exp(-self.input))
        d_activation = activation * (1 - activation)
        super().__init__(input, activation, d_activation)


# This is a child class of the Activation class that applies the ReLu Activation to its input
class ELU(Activation):

    def __init__(self, input, a=0.1, gradient=1):
        super().__init__(input)
        self.gradient, self.a = gradient, a
        activation = np.where(self.input > 0, self.input * self.gradient, self.a * (np.exp(self.input) - 1))
        d_activation = np.where(self.input <= 0, activation+a, 1)
        super().__init__(input, activation, d_activation)


# This is a child class of the Activation class that applies the ReLu Activation to its input
class Tanh(Activation):

    def __init__(self, input):
      super().__init__(input)
      activation = np.tanh(self.input)
      d_activation = 1 - activation ** 2
      super().__init__(input, activation, d_activation)


class LeakyReLu(Activation):
  def __init__(self, input, gradient=1, leak=0.01):
    super().__init__(input)
    self.gradient, self.leak = gradient, leak
    activation =  np.where(self.input > 0, self.input*self.gradient, self.input*self.leak)  
    d_activation = np.where(self.input < 0, self.leak, 1)
    super().__init__(input, activation, d_activation)
       
# This is a child class of the Activation class that applies the ReLu Activation to its input
class HardSigmoid(Activation):

    def __init__(self, input):
        super().__init__(input)
        activation = np.maximum(0, np.minimum(1, (self.input + 2) / 4))
        d_activation = 2
        super().__init__(input, activation, d_activation)


# This is a child class of the Activation class that applies the ReLu Activation to its input
class HardTanh(Activation):

    def __init__(self, input, min_value=-1, max_value=1):
        super().__init__(input)
        self.min_value, self.max_value = min_value, max_value
        activation = np.where(self.input > self.max_value, 1, np.where(self.input < self.min_value, -1, self.input))
        d_activation = 2
        super().__init__(input, activation, d_activation)


# This is a child class of the Activation class that applies the ReLu Activation to its input
class HardSwish(Activation):

    def __init__(self, input):
        super().__init__(input)
        activation = np.where(self.input <= -3, 0, np.where(self.input >= 3,
                                                            self.input, (self.input * (self.input + 3)) / 6))
        d_activation = 2
        super().__init__(input, activation, d_activation)


# This is a child class of the Activation class that applies the ReLu Activation to its input
class BinaryStep(Activation):

    def __init__(self, input):
        super().__init__(input)
        activation = np.heaviside(self.input, 1)
        d_activation = None
        super().__init__(input, activation, d_activation)


# This is a child class of the Activation class that applies the ReLu Activation to its input
class Linear(Activation):

    def __init__(self, input, gradient=2):
        super().__init__(input)
        self.gradient = gradient
        activation = np.array(self.input, dtype=float) * self.gradient
        d_activation = 1
        super().__init__(input, activation, d_activation)


# This is a child class of the Activation class that applies the ReLu Activation to its input
class Swish(Activation):

    def __init__(self, input):
        super().__init__(input)
        activation = self.input / (1 - np.exp(-self.input))
        d_activation = self.input / (1. + np.exp(-self.input)) + (1. / (1. + np.exp(-self.input))) * (1. - self.input * (1. / (1. + np.exp(-self.input))))
        super().__init__(input, activation, d_activation)


# # # # # # # # # # # # # # # # # #
# Non Linear Activation functions #
# # # # # # # # # # # # # # # # # #

class Softmax(Activation):

  def __init__(self, input):
    super().__init__(input)
    activation = np.exp(self.input)  / np.exp(self.input).sum()
    d_activation = None
    super().__init__(input, activation, d_activation)


fc1 = Softmax(-1).backward
print(fc1())
