# Link to video: https://www.youtube.com/watch?v=omz_NdFgWyU&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=6

# Softmax Activation Function
#-------------------------------------------------------------------------------------------------------------------------
"""
- We want an activation function meant for classification, and one of these is the Softmax Activation Function.
- We use Softmax Activation function because if we use ReLU to predict models, it will return either a positive or negative
number as a prediction. In this case, we would denote every negative numbers as 0, but what if ReLU returns a prediction 
of negative numbers only? If so, how can we predict the model? Well, we use Softmax for that reason.

- The backbone of the Softmax Activation function is e^x. Basically, as x approaches infinity, e^x approaches infinity
as well. On the other hand, as x approaches negative infinity, e^x approaches closer to 0.
"""
import math
import numpy as np

# We will now test with a bath of inputs
layer_outputs = [[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]]

# We assign E as the e = 2.71828182846
# E = math.e 

# Or we can use np.exp() instead. (It will automatically exponentiate all the individual values in the matrix)
exp_values = np.exp(layer_outputs)

print(exp_values)

"""
Normalization: A single output neuron value divided by the sum of all the other output neurons
"""
"""
- If axis = 0, it will return the sum of each column: [15.11, 0.451, 2.611]
- If axis = 1, it will return the sum of each row: [8.395, 7.29, 2.487]
- keepdims = True means that it will return a 3x1 matrix of the resulting sum: [[ 8.395], [ 7.29 ], [ 2.487]]
"""
norm_values = exp_values / np.sum(np.sum(layer_outputs, axis=1, keepdims=True))
print(norm_values)

# print(sum(norm_values)) # The result should add up close to 1.0

"""
Process of Softmax Activation Function: Input -> Exponentiate -> Normalize -> Output 
(Softmax Activation is the Exponentiate & Normalize process)
"""