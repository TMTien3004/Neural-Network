#Dense Layer Class
#---------------------------------------------------------------------------------------------------------------------------
import numpy as np
np.random.seed(0)

"""
- If you wish to load a pre-trained model, you will initialize the parameters to whatever that pretrained model finished with.
- When we pass data through a model from beginning to end, this is called a forward pass
"""

"""
- np.random.randn produces a Gaussian distribution with a mean of 0 and a variance of 1, which means that it will generate 
random numbers, positive and negative, centered at 0 and with the mean value close to 0.
- np.zeros function takes a desired array shape as an argument and returns an matrix filled with zeros.
"""
X = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # We'll stick with random initialization for now
        self.biases = np.zeros((1, n_neurons))
        pass # using pass statement as a placeholder

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
        pass # using pass statement as a placeholder

# Create 1 layer of 4 inputs and 5 neurons and 5 inputs and 2 neurons
layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)