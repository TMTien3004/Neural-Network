# RMSProp (Root Mean Square Propagation)
#---------------------------------------------------------------------------------------------------------------------------
"""
Similar to AdaGrad, RMSProp calculates an adaptive learning rate per parameter; it`s just calculated in a different way 
than AdaGrad.

Instead of continually adding squared gradients to a cache (like in Adagrad), it uses a moving average of the cache. Each 
update to the cache retains a part of the cache and updates it with a fraction of the new, squared, gradients.

The new hyperparameter here is rho. Rho is the cache memory decay rate.
"""
cache = rho * cache + ( 1 - rho) * gradient ** 2

# RMSProp Optimizer
class Optimizer_RMSProp:
    # Initialize the default learning rate of 1.0.
    def __init__(self, learning_rate = 1., decay = 0., epsilon = 1e-7, rho = 0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

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
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.deltaWeights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.deltaBiases ** 2

        # Vanilla SGD parameter updates + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * layer.deltaWeights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.deltaBiases / (np.sqrt(layer.bias_cache) + self.epsilon)
    
    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1
