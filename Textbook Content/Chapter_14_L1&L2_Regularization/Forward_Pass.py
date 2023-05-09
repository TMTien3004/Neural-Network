"""
Regularization methods are those which reduce generalization error. The first forms of regularization that we will address
are L1 and L2 regularization. L1 and L2 regularization are used to calculate a number (called a penalty) added to the 
loss value to penalize the model for large weights and biases.
"""

# Forward Pass
#---------------------------------------------------------------------------------------------------------------------------
"""
- L1 regularization`s penalty is the sum of all the absolute values for the weights and biases.
- L2 regularization`s penalty is the sum of the squared weights and biases.
"""
# L1 weight regularization
l1w = lambda_l1w * sum(abs(weights))
# L1 bias regularization
l1b = lambda_l1b * sum(abs(biases))
# L2 weight regularization
l2w = lambda_l2w * sum(weights ** 2)
# L2 bias regularization
l2b = lambda_l2b * sum(biases ** 2)
# Overall Loss
loss = data_loss + l1w + l1b + l2w + l2b

def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
    # Initialize weights and biases
    self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) 
    self.biases = np.zeros((1, n_neurons))

    # Set regularization strength
    self.weight_regularizer_l1 = weight_regularizer_l1
    self.weight_regularizer_l2 = weight_regularizer_l2
    self.bias_regularizer_l1 = bias_regularizer_l1
    self.bias_regularizer_l2 = bias_regularizer_l2

def regularization_loss(self, layer):
    regularization_loss = 0

    # L1 regularization - weights
    if layer.weight_regularizer_l1 > 0:
        regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

    # L2 regularization - weights
    if layer.weight_regularizer_l2 > 0:
        regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

    # L1 regularization - bias
    if layer.bias_regularizer_l1 > 0:
        regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

    # L2 regularization - bias
    if layer.bias_regularizer_l2 > 0:
        regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

    return regularization_loss

# Calculate loss from output of activation2 so softmax activation
data_loss = loss_function.forward(activation2.output, y)

# Calculate regularization penalty
regularization_loss = loss_function.regularization_loss(layer1) + loss_function.regularization_loss(layer2)

# Calculate overall loss
loss = data_loss + regularization_loss