# Tipo-Functions

This is an library of important functions used in ML and Deep Learning,
The library consist of every tool necessary for your research in AI,
everything from activation function and loss functions to complex
back propagation and easy to use forward propagation methods.


## Installation
To download the library you use the following command, because of frequent updates we recommend to use the     ```--upgrade``` syntax
```
  pip install --upgrade tipo
```

## Usage
The ```Tipo.Functional``` supports many activation and cost functions to use them simply use the command
```python
# Imports the Functional modules
From Tipo.Functional import activation

# Outputs the activation of a certain neuron
scores = activation.relu(neuron)

# Compares the score with the actual answer
loss = loss.meanSquareError(scores, answer)
```
This is just one example of all the possibilties with this library, in the above example we took the output from a network and applied an activation function we then compared the score with the actual label to get an loss that we can feed into our back propagation.

## Requirements 
- Python 3.0+

## Final Words
This library is very easy to use and will be useful for AI research and gives you more controll in comparision to other librariesIf you want to give use feedback please let us know by emailing liam_nordvall@hotmail.com

  
