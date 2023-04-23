#Dense Layer Class
#---------------------------------------------------------------------------------------------------------------------------
import numpy as np
import nnfs #Can't import nnfs
from nnfs.datasets import spiral_data_generator

nnfs.init()
"""
- If you wish to load a pre-trained model, you will initialize the parameters to whatever that pretrained model finished with.
- When we pass data through a model from beginning to end, this is called a forward pass
"""

"""
- np.random.randn produces a Gaussian distribution with a mean of 0 and a variance of 1, which means that it will generate 
random numbers, positive and negative, centered at 0 and with the mean value close to 0.
- np.zeros function takes a desired array shape as an argument and returns an matrix filled with zeros.
"""

class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # We'll stick with random initialization for now
        self.biases = np.zeros((1, n_neurons))
        pass # using pass statement as a placeholder

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        pass # using pass statement as a placeholder

#Create data set


# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense( 2 , 3 )

# Perform a forward pass of our training data through this layer


print (dense1.output[:5])