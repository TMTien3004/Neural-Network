import numpy as np

#-------------------------------------------------------------------------------------------------------------------------
# Use this as an alternative to downloading nnfs package
def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points) # radius
        t = np.linspace(class_number*4, (class_number + 1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

def vertical_data(samples, classes):
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        X[ix] = np.c_[np.random.randn(samples)*.1 + (class_number)/3, np.random.randn(samples)*.1 + 0.5]
        y[ix] = class_number
    return X, y
#-------------------------------------------------------------------------------------------------------------------------

X = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

# Dense Layer
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # We'll stick with random initialization for now
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
        # Remember input values
        self.inputs = inputs
    
    # Backward pass (Backpropagation)
    def backward(self, deltaValues):
        # Gradients on parameters
        self.deltaWeights = np.dot(self.inputs.T, deltaValues)
        self.deltaBiases = np.sum(deltaValues, axis=0, keepdims=True)
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


# Softmax Activation
class Softmax_Activation:
    def forward(self, inputs):
        # We need to subtract all values by the larget element in the input dataset to avoid overflow. The probability will not be affected by this.
        # Get max value of row, subtract all values of that row by the max value, and then you exponentiate
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Divide each value of the row with the sum of that row
        probablities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        # Return the output
        self.output = probablities

    def backward(self, deltaValues):
        # Create an uninitialized array
        self.deltaInputs = np.empty_like(deltaValues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, deltaValues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.deltaInputs[index] = np.dot(jacobian_matrix, single_dvalues)     


# Common loss class
class Loss:
    # Calculates the data and regularization losses given model output and ground truth values
    def calculate (self, output, y): # y is the intended target values
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return mean loss
        return data_loss


# Cross-Entropy Loss
class Categorical_Cross_Entropy_Loss(Loss):
    def forward(self, y_pred, y_true): # y_pred is the value from the neural network, y_true is the target training values
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        # Probabilities for target values - only if categorical labels
        # Check Neural Networks from Scratch - P.8 Implementing Loss (Skip to 11:10)
        if(len(y_true.shape) == 1): # If the target class is only 1 dimension 
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif(len(y_true.shape) == 2):
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, deltaValues, y_true):
        # Number of samples
        samples = len(deltaValues)
        # Number of labels in every sample
        labels = len(deltaValues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        # Calculate gradient
        self.deltaInputs = -y_true / deltaValues
        # Normalize gradient
        self.deltaInputs = self.deltaInputs / samples


# Softmax classifier - combined Softmax activation and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Softmax_Activation()
        self.loss = Categorical_Cross_Entropy_Loss()

    def forward(self, inputs, y_true):
        # Output layer activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
        
    def backward(self, deltaValues, y_true):
        # Number of samples
        samples = len(deltaValues)
        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)
        # Copy so we can safely modify
        self.deltaInputs = deltaValues.copy()
        # Calculate gradient
        self.deltaInputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.deltaInputs = self.deltaInputs / samples

# Stochastic Gradient Decent (SGD) Optimizer
class Optimizer_SGD:
    # Initialize the default learning rate of 1.0.
    def __init__(self, learning_rate = 1., decay = 0., momentum = 0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):
        if(self.decay):
            self.current_learning_rate = self.learning_rate * (1. / (1 + self.decay * self.iterations))

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
    
    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# We set up 100 feature sets of 3 classes
X, y = spiral_data(points=100, classes=3)

# Create first dense layer with 2 input features and 64 output values
layer1 = Layer_Dense(2, 64)

# Create ReLU Activation
activation1 = ReLU_Activation()

# Create second dense layer with 64 input features and 3 output values
layer2 = Layer_Dense(64, 3)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create optimizer object with decay rate = 0.001 and momentum = 0.9
# (There are parameters that might return better results, so set decay in anyway you want and have fun experimenting!)
optimizer = Optimizer_SGD(decay=1e-3, momentum=0.9)

# Train in loop
for epoch in range (10001):
    # Perform a forward pass of the first layer
    layer1.forward(X)

    # Perform a forward pass through activation function
    activation1.forward(layer1.output)

    # Perform a forward pass of the second layer and takes outputs of activation function of first layer as inputs
    layer2.forward(activation1.output)

    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss = loss_activation.forward(layer2.output, y)

    # Calculate accuracy from output of activation2 and targets calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis = 1)
    if len (y.shape) == 2 :
        y = np.argmax(y, axis = 1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print("epoch: {}, acc: {:.3f}, loss: {:.3f}, lr: {}".format(epoch, accuracy, loss, optimizer.current_learning_rate))

    # Backward pass (Backpropagation)
    loss_activation.backward(loss_activation.output, y)
    layer2.backward(loss_activation.deltaInputs)
    activation1.backward(layer2.deltaInputs)
    layer1.backward(activation1.deltaInputs)

    # Update the weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)
    optimizer.post_update_params()

# Print gradients
# print (layer1.deltaWeights)
# print (layer1.deltaBiases)
# print (layer2.deltaWeights)
# print (layer2.deltaBiases)