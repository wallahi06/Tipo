import numpy as np


# This is the base class for all the loss and cost functions



# This is a child class of the loss class, it calculates the MSE Loss
class meanSquareError():

    def __call__(self, true_value, prediction):
        self.true_value = true_value
        self.prediction = prediction
        return np.mean(np.power(self.true_value - self.prediction, 2))

    def backward(self):
        return 2 * (self.prediction - self.true_value) / np.size(self.true_value)


# This is a child class of the loss class, it calculates the MAE Loss
class meanAbsoluteError():

    def __call__(self, true_value, prediction):
        self.true_value = true_value
        self.prediction = prediction
        return np.mean(np.abs(self.true_value - self.prediction))

    def backward(self):
        pass


# This is a child class of the loss class, it calculates the MBE Loss
class meanBiasError():

    def __call__(self, true_value, prediction):
        self.true_value = true_value
        self.prediction = prediction
        return (self.true_value - self.prediction).mean()

    def backward(self):
        pass



# Linear Activation Functions
# This class applies the ReLu Activation to its input
class ReLu():

    def __init__(self, inputs, gradient=1):
      self.gradient = gradient
      self.inputs = np.array(inputs)

    def __call__(self):
      return np.where(self.inputs > 0, self.inputs * self.gradient, 0)

    def prime(self):
      return (self.inputs > 0) * 1


# This class applies the Sigmoid Activation to its input
class Sigmoid():

    def __init__(self, inputs, gradient=1):
      self.inputs = np.array(inputs)
      self.activation = 1 / (1 + np.exp(-self.inputs))

    def __call__(self):
      return self.activation

    def prime(self):
      return self.activation * (1 - 1 / self.activation)


# This class applies the ELU Activation to its input
class ELU():

    def __init__(self, inputs, a=0.1, gradient=1):
      self.inputs = np.array(inputs)
      self.gradient, self.a = gradient, a
      self.activation = np.where(self.inputs > 0, self.inputs * self.gradient, self.a * (np.exp(self.inputs) - 1))

    def __call__(self):
      return self.activation

    def prime(self):
      return np.where(self.inputs <= 0, self.activation+self.a, 1)


# This class applies the Tanh Activation to its input
class Tanh():

    def __init__(self, inputs):
      self.inputs = np.array(inputs)
      self.activation = np.tanh(self.inputs)

    def __call__(self):
      return self.activation

    def prime(self):
      return 1 - self.activation ** 2


# This class applies the Leaky ReLu Activation to its input
class LeakyReLu():

    def __init__(self, inputs, gradient=1, leak=0.01):
      self.inputs = np.array(inputs)
      self.gradient, self.leak = gradient, leak

    def __call__(self):
      return np.where(self.inputs > 0, self.inputs*self.gradient, self.inputs*self.leak)  
    
    def prime(self):
      return np.where(self.inputs < 0, self.leak, 1)
       

# This class applies the Hard Sigmoid Activation to its input
class HardSigmoid():

    def __init__(self, inputs):
      self.inputs = np.array(inputs)

    def __call__(self):
      return np.maximum(0, np.minimum(1, (self.inputs + 2) / 4))
    
    def prime(self):
      return None


# This class applies the Hard Tanh Activation to its input
class HardTanh():

    def __init__(self, inputs, min_value=-1, max_value=1):
      self.inputs = np.array(inputs)
      self.min_value, self.max_value = min_value, max_value

    def __call__(self):
      return np.where(self.inputs > self.max_value, 1, np.where(self.inputs < self.min_value, -1, self.inputs))

    def prime(self):
      return None


# This class applies the Hard Swish Activation to its input
class HardSwish():

    def __init__(self, inputs):
      self.inputs = np.array(inputs)
    
    def __call__(self):
      return np.where(self.inputs <= -3, 0, np.where(self.inputs >= 3,
                                                    self.inputs, (self.inputs * (self.inputs + 3)) / 6))
    
    def prime(self):
      return None


# This class applies the Binary Step Activation to its input
class BinaryStep():

    def __init__(self, inputs):
      self.inputs = np.array(inputs)
    
    def __call__(self):
      return np.heaviside(self.inputs, 1)

    def prime(self):
      return None


# This class applies the Linear Activation to its input
class Linear():

    def __init__(self, inputs, gradient=2):
      self.inputs = np.array(inputs)
      self.gradient = gradient

    def __call__(self):
      return np.array(self.inputs, dtype=float) * self.gradient

    def prime(self):
      return 1


# This class applies the Swish Activation to its input
class Swish():

    def __init__(self, inputs):
      self.inputs = np.array(inputs)

    def __call__(self):
      return self.inputs / (1 - np.exp(-self.inputs))

    def prime(self):
      return self.inputs / (1 + np.exp(-self.inputs)) + (1. / (1. + np.exp(-self.inputs))) * (1. - self.inputs * (1. / (1. + np.exp(-self.inputs))))


# Non Linear Activation functions 
# This class that applies the Softmax Activation to its input
class Softmax():

    def __init__(self, inputs):
      self.inputs = np.array(inputs)

    def __call__(self):
      return np.exp(self.inputs)  / np.exp(self.inputs).sum()

    def prime(self):
      return None

