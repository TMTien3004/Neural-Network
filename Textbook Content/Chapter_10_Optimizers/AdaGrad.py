# AdaGrad (Adaptive Gradient)
#---------------------------------------------------------------------------------------------------------------------------
"""
The idea here is to normalize updates made to the features. During the training process, some weights can rise 
significantly, while others tend to not change by much. 

AdaGrad provides a way to normalize parameter updates by keeping a history of previous updates — the bigger the sum of 
the updates is, in either direction (positive or negative), the smaller updates are made further in training. This lets
less-frequently updated parameters to keep-up with changes, effectively utilizing more neurons for training.

The cache holds a history of squared gradients, and the parm_updates is a function of the learning rate multiplied by the 
gradient (basic SGD so far) and then is divided by the square root of the cache plus some epsilon value.
"""
cache += parm_gradient ** 2
parm_updates = learning_rate * parm_gradient / (sqrt(cache) * eps)

import numpy as np

# Adagrad Optimizer
class Optimizer_Adagrad:
    # Initialize the default learning rate of 1.0.
    def __init__(self, learning_rate = 1., decay = 0., epsilon = 1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before any parameter updates
    def pre_update_params(self):
        if(self.decay):
            self.current_learning_rate = self.learning_rate * (1. / (1 + self.decay * self.iterations))

    # Update paramaters
    def update_params(self, layer):
        # If layer does not contain cache arrays, create and fill them with zeros
        if not hasattr(layer, 'weight_cache'):
            # If there is no weight, that also means there is no bias with momentum.
            # So we add momentum to biases as well
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            
        # Update cache with squared current gradients
        layer.weight_cache += layer.deltaWeights**2
        layer.bias_cache += layer.deltaBiases**2

        # Vanilla SGD parameter updates + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * layer.deltaWeights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.deltaBiases / (np.sqrt(layer.bias_cache) + self.epsilon)
        
        # Update weights and biases using either vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates
    
    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1