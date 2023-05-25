# Model Object
#---------------------------------------------------------------------------------------------------------------------------
"""
For the sake of clarity, we will create a model object that will hold all the layers and the loss and optimizer functions. 
"""

import numpy as np

def sine_data(samples=1000):
    X = np.arange(samples).reshape(-1, 1) / samples
    y = np.sin(2 * np.pi * X).reshape(-1, 1)
    return X, y

# Create dataset
X, y = sine_data()

# Dense Layer
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1, n_neurons))

        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
    
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

# ReLU Activation
class ReLU_Activation:
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)
    
    # Backward pass (Backpropagation)
    def backward(self, deltaValues):
        # Gradients on values
        self.deltaInputs = deltaValues.copy()

        # Zero gradient where input values were negative
        self.deltaInputs[self.inputs <= 0 ] = 0

# Linear Activation
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

# Model class
class Model:
    def __init__(self):
        # Create a list of network objects
        self.layers = []
    
    # Add a objects to the model
    def add(self, layer):
        self.layers.append(layer)


    # Set loss, optimizer and accuracy
    def set(self, *, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer


    # Finalize the model
    def finalize(self):
        # Create and set the input layer
        self.inputLayer = Layer_Input()
    
        # Count all the objects
        layer_count = len(self.layers)
        
        # Iterate the objects
        for i in range(layer_count):

            # If it's the first layer, the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.inputLayer
                self.layers[i].next = self.layers[i+1]
            
            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            
            # The last layer - the next object is the loss
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss

            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            # add it to the list of trainable layers
            # We don't need to check for biases -
            # checking for weights is enough
            if hasattr (self.layers[i], 'weights'):
            self.trainable_layers.append(self.layers[i])
            
                
    # Train the model
    def train(self, X, y, *, epochs=100, print_every=1000):
        # Main training loop
            for epoch in range ( 1 , epochs + 1 ):
                # Perform the forward pass
                output = self.forward(X)

                # Temporary
                print (output)
                exit()

    # Forward pass
    def forward(self, X):
        # Call forward method on the input layer
        self.input_layer.forward(X)

        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output)
        
        # Return output
        return layer.output

class Layer_Input:
    # Forward pass
    def forward(self, inputs):
        self.output = inputs

# Create model
model = Model()
# Add layers
model.add(Layer_Dense(1 , 64))
model.add(ReLU_Activation())
model.add(Layer_Dense(64 , 64))
model.add(ReLU_Activation())
model.add(Layer_Dense(64, 1))
model.add(Activation_Linear())

# Set loss and optimizer objects
model.set(loss=Mean_Squared_Error_Loss(), optimizer=Optimizer_Adam(decay=1e-3))

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, epochs=10000, print_every=100)