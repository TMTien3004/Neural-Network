import os
import numpy as np

#A Layer of Neurons with NumPy
#-------------------------------------------------------------------------------------------------------------------------
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
bias = [2.0, 3.0, 0.5]

#np.dot represents the dot product function in NumPy
layer_outputs = np.dot(weights, inputs) + bias

"""
Visualization:
layer_outputs = [np.dot(weights[0], inputs), np.dot(weights[1], inputs), np.dot(weights[2], inputs)]
"""

os.system("clear")
print(layer_outputs)