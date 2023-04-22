import numpy as np

#Transposition for the Matrix Product
#-------------------------------------------------------------------------------------------------------------------------
a = [1.0, 2.0, 3.0]
b = [2.0, 3.0, 4.0]

#np.expand_dims() adds a new dimension at the index of the axis .
c = np.expand_dims(np.array(a), axis = 0)
print(c)

#.T means that we will turn matrix b into the TRANSPOSE of matrix b
a = np.array([a])
b = np.array([b]).T
print(np.dot(a,b))

"""
The reason why we need to do the transpose is because if we want to do matrix multiplication, we need to make sure that
the column of matrix a matches with the row of matrix b (3x4 & 4x3). Since matrix b originally is a 3x4, the transpose 
will make sure the matrix multiplication works.
"""

inputs = [[1.0, 2.0, 3.0, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

# First, we do matrix multiplication, and then we add up each row in the matrix with the biases.
outputs = np.dot(inputs, np.array(weights).T) + biases
print(outputs)
