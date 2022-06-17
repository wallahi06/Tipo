# This code is under MIT License and free to use as it is, the code have been written by Liam Nordvall 
# togheter with Lord Hummer. The module includes most of the functions needed for everytinh from basic
# Neural Networks to more deep networks with many layers and nodes.


import numpy as np


# This class consists of all the necessary activation functions needed, we have put them into different
# methods to make it easy to use togheter with other modules. We made the methods ready to use but also 
# added additional parameters for our professional users.

class ActivationFunctions:

    def __init__(self):
        pass

    def binarystep(value):
        value = np.array(value)
        return np.heaviside(value, 1)

    def linear(self, value, gradient=4):
        value = np.array(value)
        return gradient*value

    def sigmoid(self, value):
        value = np.array(value)
        return 1/(1 + np.exp(-value))

    def tanh(self, value):
        value = np.array(value)
        return 2/(1 + np.exp(-2*value)) - 1

    def relu(self, value, gradient=1):
        value = np.array(value)
        if value.all() < 0:
            return 0
        else:
            return value*gradient

    def leakyrelu(self, value, gradient=1, leak=0.01):
        value = np.array(value)
        if value < 0:
            return value*leak
        else:
            return value*gradient

    def elu(self, value, a=0.1):
        value = np.array(value)
        if value < 0:
            return a*(np.exp(value)-1)
        else:
            return value

    def swish(self, value):
        value = np.array(value)
        return value/(1-np.exp(-value))

    def softmax(self, value):
        z = np.exp(value)
        value_ = z / z.sum()
        return value_


# We initilize the class with a keyword for easier use, namely, instead of writing activation = ActivationFunctions() 
# you just import activation class from Tipo.Functional to get access to all the activation functions

activation = ActivationFunctions()


# This class includes all the important Loss and Cost functions, we have made the use of these funcitons
# much easier by naming the methods to relevant names and using few parameters but at the same time giving more parameters for
# fine tuning to our professional users.

class LossFunctions:

    def __init__(self):
        pass

    def meanAbsoluteError(self, true_value, prediction):
        true_value, prediction = np.array(true_value), np.array(prediction)
        return np.mean(np.abs(true_value - prediction))

    def meanSquareError(self, true_value, prediction):
        true_value, prediction = np.array(true_value), np.array(prediction)
        return np.square(true_value - prediction).mean()


loss = LossFunctions()

