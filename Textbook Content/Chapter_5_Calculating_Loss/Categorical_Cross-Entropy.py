# Link to videos: 
# https://www.youtube.com/watch?v=dEXPMQXoiLc&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=7
# https://www.youtube.com/watch?v=levekYbxauw&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=8


# Categorical Cross-Entropy Loss
#-------------------------------------------------------------------------------------------------------------------------
"""
- The purpose of Categorical Cross-Entropy Loss is to determine how wrong the model is.
- One-hot encoding:
+ Assume that you have a vector that is n-classes long (a vector of length n) and that vector is filled with zeros, except
the index of the target class has a 1. 
+ All of the log(n) we use in here are considered ln(n)
"""
import math
import numpy as np

# Set up an sample output value for softmax activation function
softmax_output = [0.7, 0.1, 0.2] 

# Assume that the zero-th index of the one-hot vector is "hot" (the value is 1) and the rest is zero.
target_value = [1, 0, 0]

# The higher the rate of prediction is, the loss is lower and vice versa.
loss = 0
for i in range(len(target_value)):
    loss += math.log(softmax_output[i]) * target_value[i]
loss *= -1
print(loss)