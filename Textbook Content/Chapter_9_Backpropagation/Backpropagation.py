# Backpropagation
#-------------------------------------------------------------------------------------------------------------------------
import math
import numpy as np

"""
Backpropagation is a process of using the chain rule to get the resulting output functions gradients, and they are 
passed back through the neural network, using multiplication of the gradient of subsequent functions from later 
layers with the current one.

The first step is to backpropagate our gradients by calculating derivatives and partial derivatives with respect to 
each of our parameters and inputs. To do this, we are going to use the chain rule.
"""
# Forward pass
inputs = [1.0, -2.0, 3.0]
weights = [-3.0, -1.0, 2.0] 
bias = 1.0

# Multiplying inputs by weights
xw0 = inputs[0] * weights[0]
xw1 = inputs[1] * weights[1]
xw2 = inputs[2] * weights[2]

# Adding weighted inputs and a bias
z = xw0 + xw1 + xw2 + bias

# ReLU activation function
y = max (z, 0)

# Backward pass
# The derivative from the next layer
dvalue = 1.0 # Assume that our neuron receives a gradient of 1

# Derivative of ReLU and the chain rule
drelu_dz = dvalue * (1. if z > 0 else 0.)
print(drelu_dz)

"""
We work on the following partial derivatives:
- drelu_dxw0 as the partial derivative of the ReLU w.r.t. the first weighed input, w0x0
- drelu_dxw1 as the partial derivative of the ReLU w.r.t. the second weighed input, w1x1
- drelu_dxw2 as the partial derivative of the ReLU w.r.t. the third weighed input, w2x2
- drelu_db as the partial derivative of the ReLU with respect to the bias, b.
"""

# Partial derivatives of the multiplication, the chain rule
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
drelu_db = 1
drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_db * drelu_db
print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

dmul_dx0 = weights[0]
dmul_dx1 = weights[1]
dmul_dx2 = weights[2]
dmul_dw0 = inputs[0]
dmul_dw1 = inputs[1]
dmul_dw2 = inputs[2]
drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw2 = drelu_dxw2 * dmul_dw2
print (drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)