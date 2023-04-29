# Backpropagation
#-------------------------------------------------------------------------------------------------------------------------
import math
import numpy as np

# Passed in gradient from the next layer (In this case, we are going to use an array of incremental gradient values)
deltaValues = np.array([[1., 1., 1.], [2., 2., 2.],[3., 3., 3.]])

# We have 3 sets of inputs - samples
inputs = np.array([[1, 2, 3, 2.5], [2., 5., -1., 2],[-1.5, 2.7, 3.3, -0.8]])

# Create 3 sets of weights - one set for each neuron, and each set has four inputs. (Don't forget that it is transposed)
weights = np.array([[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]).T

# One bias for each neuron
# Biases are the row vector with a shape (1, neurons)
biases = np.array([[2, 3, 0.5]])

# Forward pass
layer_outputs = np.dot(inputs, weights) + biases # Dense layer
relu_outputs = np.maximum(0, layer_outputs) # ReLU Activation

#ReLU activation's derivative
deltaRelu = relu_outputs.copy()
deltaRelu[layer_outputs <= 0] = 0

# Dense Layer
# Gradient of the neuron is simply the dot product of each weights with the input gradient.
deltaInputs = np.dot(deltaRelu, weights.T)

# Gradient of the weight is simply the dot product of the input with the Relu activation's derivative.
deltaWeights = np.dot(inputs.T, deltaRelu)

# Gradient of the biases (dbiases - sum values, do this over samples (first axis), keepdims)
deltaBiases = np.sum(deltaRelu, axis=0, keepdims=True)

#Update parameters
weights += -0.001 * deltaWeights
biases += -0.001 * deltaBiases

print(deltaInputs)
print(deltaBiases)

# Gradient of the ReLU Activation Function
#-------------------------------------------------------------------------------------------------------------------------
# Example layer output
z = np.array([[1, 2, -3, -4],[2, -7, -1, 3],[-1, 2, 5, -1]])
dvalues = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

"""
To calculate the ReLU derivative, we create an array filled with zeros by using np.zeros_like. Then, we set the values
related to the output greater than 0 is 1. 
"""
# ReLU activation's derivative
drelu = np.zeros_like(z)
drelu[z > 0] = 1
# print(drelu)

# The chain rule
drelu *= dvalues
# print (drelu)