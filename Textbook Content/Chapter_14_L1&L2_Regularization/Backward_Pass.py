# Backward Pass
#---------------------------------------------------------------------------------------------------------------------------
# Dense Layer
class Layer_Dense:
    # Backward pass (Backpropagation)
    def backward(self, deltaValues):
        # Gradients on parameters
        self.deltaWeights = np.dot(self.inputs.T, deltaValues)
        self.deltaBiases = np.sum(deltaValues, axis=0, keepdims=True)
        
        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.deltaWeights += self.weight_regularizer_l1 * dL1
        
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.deltaWeights += 2 * self.weight_regularizer_l2 * self.weights
        
        # L1 on bias
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.deltaBiases += self.bias_regularizer_l1 * dL1
        
        # L2 on bias
        if self.bias_regularizer_l2 > 0:
            self.deltaBiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradients on values
        self.deltaInputs = np.dot(deltaValues, self.weights.T)

print("epoch: {}, acc: {:.3f}, loss: {:.3f}, data_loss: {:.3f}, reg_loss: {:.3f}, lr: {}".format(epoch, accuracy, loss, data_loss, regularization_loss, optimizer.current_learning_rate))
# Create Dense layer with 2 input features and 3 output values
layer1 = Layer_Dense(2, 64, weight_regularizer_l2 = 5e-4, bias_regularizer_l2 = 5e-4)