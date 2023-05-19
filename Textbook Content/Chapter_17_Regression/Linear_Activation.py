# Linear Activation
#---------------------------------------------------------------------------------------------------------------------------
"""
Since we are not working with classification labels anymore, we cannot calculate cross-entropy. Instead, we need some new 
methods. The two main methods for calculating error in regression are mean squared error (MSE) and mean absolute error (MAE).
"""

# Linear activation
class Activation_Linear:
    # Forward pass
    def forward(self, inputs):
        # Just remember values
        self.inputs = inputs
        self.output = inputs

    # Backward pass
    def backward(self, deltaValues):
        # Derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.deltaInputs = deltaValues.copy()