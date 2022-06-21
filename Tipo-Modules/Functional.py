# Import all the necessary libraries
import numpy as np
from tipo.modules import Node


# This is class calculates the MSE Loss
class meanSquareError():

    def __call__(self, true_value, prediction):
        self.true_value = true_value
        self.prediction = prediction
        return np.mean(np.power(self.true_value - self.prediction, 2))

    def backward(self):
        return 2 * (self.prediction - self.true_value) / np.size(self.true_value)


# This is class calculates the MAE Loss
class meanAbsoluteError():

    def __call__(self, true_value, prediction):
        self.true_value = true_value
        self.prediction = prediction
        return np.mean(np.abs(self.true_value - self.prediction))

    def backward(self):
        pass


# This is class calculates the MBE Loss
class meanBiasError():

    def __call__(self, true_value, prediction):
        self.true_value = true_value
        self.prediction = prediction
        return (self.true_value - self.prediction).mean()

    def backward(self):
        pass


# This is the parent class of all the activation classes
class Activation(Node):
  def __init__(self, activation, d_activation):
    self.activation = activation
    self.d_activation = d_activation

  def forward(self, input):
    self.input = np.array(input)
    return self.activation(self.input)

  def backward(self, output_gradient, lr):
    pass


# This class applies the Tanh Activation to its input
class Tanh(Activation):
  def __init__(self):
    self.tanh = lambda x: np.tanh(x)
    self.d_tanh = lambda x: 1 - np.tanh(x) ** 2
    super().__init__(self.tanh, self.d_tanh)


# This class applies the ReLu Activation to its input
class Relu(Activation):
  def __init__(self, gradient=1):
    self.gradient = gradient
    self.relu = lambda x: np.where(x > 0, x * self.gradient, 0)
    self.d_relu = lambda x: (x > 0) * 1
    super().__init__(self.relu, self.d_relu)


# This class applies the Sigmoid Activation to its input
class Sigmoid(Activation):
  def __init__(self, gradient=1):
    self.gradient = gradient
    self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
    self.d_sigmoid = lambda x: self.sigmoid * (1 - 1 / self.sigmoid)
    super().__init__(self.sigmoid, self.d_sigmoid)


 # This class applies the ELU Activation to its input
class ELU(Activation):
  def __init__(self, a=0.1, gradient=1):
    self.gradient, self.a = gradient, a
    self.elu = lambda x: np.where(x > 0, x * self.gradient, self.a * (np.exp(x) - 1))
    self.d_elu = lambda x: np.where(x <= 0, self.elu+self.a, 1)
    super().__init__(self.elu, self.d_elu)


# This class applies the Leaky ReLu Activation to its input
class LeakyRelu(Activation):
  def __init__(self, gradient=1, leak=0.01):
    self.gradient, self.leak = gradient, leak
    self.leakyrelu = lambda x: np.where(x > 0, x*self.gradient, x*self.leak)  
    self.d_leakyrelu = lambda x: np.where(x < 0, self.leak, 1)
    super().__init__(self.leakyrelu, self.d_leakyrelu)


# This class applies the Hard Sigmoid Activation to its input
class HardSigmoid(Activation):
  def __init__(self):
    self.hardsigmoid = lambda x: np.maximum(0, np.minimum(1, (x + 2) / 4))
    self.d_hardsigmoid = lambda x: None
    super().__init__(self.hardsigmoid, self.d_hardsigmoid)


 # This class applies the Hard Tanh Activation to its input
class HardTanh(Activation):
  def __init__(self, min_value=-1, max_value=1):
    self.min_value, self.max_value = min_value, max_value
    self.hardtanh = lambda x: np.where(x > self.max_value, 1, np.where(x < self.min_value, -1, x))   
    self.d_hardtanh = lambda x: None 
    super().__init__(self.hardtanh, self.d_hardtanh)


# This class applies the Hard Swish Activation to its input
class HardSwish(Activation):
  def __init__(self):
    self.hardswish = lambda x: np.where(x <= -3, 0, np.where(x >= 3, x, (x * (x + 3)) / 6))
    self.d_hardswish = lambda x: None
    super().__init__(self.hardswish, self.d_hardswish)


# This class applies the Binary Step Activation to its input
class BinaryStep(Activation):
  def __init__(self):
    self.binarystep = lambda x: np.heaviside(x, 1)
    self.d_binarystep = lambda x: None
    super().__init__(self.binarystep, self.d_binarystep)


# This class applies the Linear Activation to its input
class Linear(Activation):
  def __init__(self, gradient=2):
    self.gradient = gradient
    self.linear = lambda x: x * self.gradient
    self.d_linear = lambda x: 1
    super().__init__(self.linear, self.d_linear)


# This class applies the Swish Activation to its input
class Swish(Activation):
  def __init__(self):
    self.swish = lambda x: x / (1 - np.exp(-x))
    self.d_swish = lambda x: x / (1 + np.exp(-x)) + (1. / (1. + np.exp(-x))) * (1. - x * (1. / (1. + np.exp(-x))))
    super().__init__(self.swish, self.d_swish)
