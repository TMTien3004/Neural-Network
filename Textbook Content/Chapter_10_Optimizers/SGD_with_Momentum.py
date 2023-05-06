# Stochastic Gradient Decent (SGD) with momentum
#---------------------------------------------------------------------------------------------------------------------------
"""
- Momentum uses the previous update`s direction to influence the next update`s direction, minimizing the chances of bouncing 
around and getting stuck at the local minimum.
"""
# Update paramaters
def update_params(self, layer):
    # If we use momentum
    if self.momentum:
        # If layer does not contain momentum arrays, create and fill them with zeros
        if not hasattr(layer, 'weight_momentums'):
            # If there is no weight, that also means there is no bias with momentum.
            # So we add momentum to biases as well
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
        
        # If layer has momentum, we update weight and biases.
        weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.deltaWeights
        layer.weight_momentums = weight_updates

        # Update bias with momentum
        bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.deltaBiases
        layer.bias_momentums = bias_updates

    # Vanilla SGD updates (as before momentum update)
    else:
        layer.weights += -self.learning_rate * layer.deltaWeights
        layer.biases += -self.learning_rate * layer.deltaBiases
    
    # Update weights and biases using either vanilla or momentum updates
    layer.weights += weight_updates
    layer.biases += bias_updates