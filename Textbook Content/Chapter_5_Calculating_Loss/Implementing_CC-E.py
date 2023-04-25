# Link to videos: 
# https://www.youtube.com/watch?v=levekYbxauw&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=8


# Categorical Cross-Entropy Loss
#-------------------------------------------------------------------------------------------------------------------------
import math
import numpy as np

# Set up an sample output value for softmax activation function and the target value
softmax_outputs = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]])
class_targets = [0, 1, 1]

# We want to return an array of the softmax output values in each subarray with its respective index based on the value of the target value 
# for target_idx, distribution in zip (class_targets, softmax_outputs):
#     print(distribution[target_idx])

# We can use the NumPy method
"""
To illustrate this, we look at the class_targets vector. At idx 0, it has a value of 0. Hence, we return the value at the 
0-th row and 0-th column. Similarly, at idx 1, it has a value of 1, so we return the value at the 1st row and 1st column.
"""
neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
average_loss = np.mean(neg_log)
print(average_loss)
print(-np.log(softmax_outputs[[0, 1, 2], [class_targets]]))

# What if neg_log is 0 (aka one of the values in the softmax_outputs is 0.0)?
# => We clip the values in the range by some fairly insignificant amount 