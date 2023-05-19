# Mean Absolute Error
#---------------------------------------------------------------------------------------------------------------------------
"""
Mean Absolute Error (MAE): you take the absolute difference between the predicted and true values in a single output and
average those absolute values.
"""
import numpy as np

# Common loss class
class Loss:
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

    # Calculates the data and regularization losses given model output and ground truth values
    def calculate (self, output, y): # y is the intended target values
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return mean loss
        return data_loss

# Mean Absolute Error Loss
class Mean_Absolute_Error_Loss(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis= -1)
        return sample_losses
    
    def backward(self, deltaValues, y_true):
        # Number of samples
        samples = len(deltaValues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(deltaValues[0])

        # Calculate gradient
        self.deltaInputs = np.sign(y_true - deltaValues) / outputs
        # Normalize gradient
        self.deltaInputs = self.deltaInputs / samples