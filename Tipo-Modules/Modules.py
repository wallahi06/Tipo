import numpy as np


class Node(object):
  def __init__(self):
    self.input = None
    self.output = None

  def forward(self, input):
    pass

  def backward(self, output_gradient, lr):
    pass



class DenseLayer(Node):
  def __init__(self, input_size, output_size, bias_enabled=True):
    self.bias_enabled = bias_enabled
    self.weights = np.random.randn(output_size, input_size)
    self.biases = np.where(bias_enabled==True, np.random.randn(1, output_size), np.zeros((1, output_size)))
    
  
  def forward(self, input):
    self.input = input
    output = np.dot(self.weights, self.input) + self.biases
    return output


  def backward(self, output_gradient, learning_rate):

    # Calculate the adjustment of the weights
    weights_gradient = np.dot(output_gradient, self.input)

    # Update the weights
    self.weights -= learning_rate * weights_gradient

    # Only update the bias if bias_enabled == True
    if self.bias_enabled == True:
      self.biases -= learning_rate * output_gradient
    
    return np.dot(self.weights.T, output_gradient)
