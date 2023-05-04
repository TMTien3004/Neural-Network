# Softmax Activation Derivative Example
#---------------------------------------------------------------------------------------------------------------------------
"""
- np.eye(): Creates a an identity matrix according to the size of the vector input.
- np.diagflat(): creates an array using an input vector as the diagonal.

Jacobian Matrix: 
- An array of partial derivatives in all of the combinations of both input vectors.
"""
import numpy as np

#Sample Softmax Output
softmax_output = [0.7 , 0.1 , 0.2]

#Reshape as a list of samples
softmax_output = np.array(softmax_output).reshape(-1, 1)

# Create an array using an input vector as the diagonal.
# print(softmax_output * np.eye(softmax_output.shape[0]))
# (We can replace it with np.diagflat())
print(np.diagflat(softmax_output))

# Form the Jacobian Matrix by taking diagflat() subtract by the dot product of the output by itself
Jacobian_mat = np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T)
print(Jacobian_mat)


