import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------------------------------------
# Use this as an alternative to downloading nnfs package
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

def sine_data(samples=1000):
    X = np.arange(samples).reshape(-1, 1) / samples
    y = np.sin(2 * np.pi * X).reshape(-1, 1)
    return X, y
#-------------------------------------------------------------------------------------------------------------------------

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


# Dropout
class Layer_Dropout:
    def __init__(self, rate):
        # Store rate (how much percentage of neurons we want to keep back)
        self.rate = 1 - rate
    
    # Forward pass
    def forward(self, inputs):
        # Save input values 
        self.inputs = inputs
        # Generate and save scale mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, deltaValues):
        # Gradient on values
        self.deltaInputs = deltaValues * self.binary_mask


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
    
    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs


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

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis = 1)    


# Sigmoid Activation Function
class Activation_Sigmoid:
    def forward(self, inputs):
        # Save input and calculate/save output
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, deltaValues):
        # Derivative - calculates from output of the sigmoid function
        self.deltaInputs = deltaValues * (1 - self.output) * self.output

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return (outputs > 0.5 ) * 1


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

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs


# Common loss class
class Loss:
    # Regularization loss calculation
    def regularization_loss(self, layer):
        regularization_loss = 0

        # Calculate regularization loss
        # iterate all trainable layers
        for layer in self.trainable_layers:
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

    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    # Calculates the data and regularization losses given model output and ground truth values
    def calculate (self, output, y): # y is the intended target values
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return mean loss
        return data_loss, self.regularization_loss()


# Model class
class Model:
    def __init__(self):
        # Create a list of network objects
        self.layers = []
    
    # Add a objects to the model
    def add(self, layer):
        self.layers.append(layer)


    # Set loss, optimizer and accuracy
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy


    # Finalize the model
    def finalize(self):
        # Create and set the input layer
        self.inputLayer = Layer_Input()
    
        # Count all the objects
        layer_count = len(self.layers)

        # Initialize a list containing trainable layers:
        self.trainable_layers = []
        
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
                self.output_layer_activation = self.layers[i]

            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            # add it to the list of trainable layers
            # We don't need to check for biases -
            # checking for weights is enough
            if hasattr (self.layers[i], 'weights'):
            self.trainable_layers.append(self.layers[i])
        
        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)
                
    # Train the model
    def train(self, X, y, *, epochs=1, print_every=1):
        # Initialize accuracy object
        self.accuracy.init(y)

        # Main training loop
            for epoch in range ( 1 , epochs + 1 ):
                # Perform the forward pass
                output = self.forward(X)
                
                # Calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, y)
                loss = data_loss + regularization_loss

                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, y)

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


# Common accuracy class
class Accuracy:
    # Calculates an accuracy
    # given predictions and ground truth values
    def calculate(self, predictions, y):

        # Get comparison results
        comparisons = self.compare(predictions, y)

        # Calculate an accuracy
        accuracy = np.mean(comparisons)

        return accuracy


# Accuracy calculation for regression model
class Accuracy_Regression(Accuracy):
    
    def __init__(self):
        # Create precision property
        self.precision = None

    # Calculates precision value
    # based on passed in ground truth
    def init(self, y, reinit = False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
    
    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision


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


# Binary Cross-Entropy Loss
class Binary_Cross_Entropy_Loss(Loss):
    def forward(self, y_pred, y_true):
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis = -1)

        # Return sample loss
        return sample_losses

    def backward(self, deltaValues, y_true):
        # Number of samples
        samples = len(deltaValues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(deltaValues[0])

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_deltaValues = np.clip(deltaValues, 1e-7 , 1 - 1e-7)

        # Calculate gradient
        self.deltaInputs = -(y_true / clipped_deltaValues - (1 - y_true) / (1 - clipped_deltaValues)) / outputs
        
        # Normalize gradient
        self.deltaInputs = self.deltaInputs / samples


# Mean Squared Error Loss
class Mean_Squared_Error_Loss(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean((y_true - y_pred) ** 2 , axis= -1)
        return sample_losses
    
    def backward(self, deltaValues, y_true):
        # Number of samples
        samples = len(deltaValues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(deltaValues[0])
        
        # Gradient on values
        self.deltaInputs = -2 * (y_true - deltaValues) / outputs
        # Normalize gradient
        self.deltaInputs = self.deltaInputs / samples


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
    
    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


# RMSProp Optimizer
class Optimizer_RMSProp:
    # Initialize the default learning rate of 1.0.
    def __init__(self, learning_rate = 0.001, decay = 0., epsilon = 1e-7, rho = 0.9):
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


# Adam (Adaptive Momentum) Optimizer
class Optimizer_Adam:
    # Initialize the default learning rate of 1.0.
    def __init__(self, learning_rate = 0.001, decay = 0., epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if(self.decay):
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update paramaters
    def update_params(self, layer):
        # If layer does not contain cache arrays, create and fill them with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.deltaWeights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.deltaBiases

        # Get corrected momentum 
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1 ))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1 ))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.deltaWeights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.deltaBiases**2

        # Get corrected cache  
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter updates + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
    
    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


# Create dataset
X, y = sine_data()

# Create first dense layer with 1 input features and 64 output values
layer1 = Layer_Dense(1, 64)

# Create ReLU Activation
activation1 = ReLU_Activation()

# Create second dense layer with 64 input features and 64 output values
layer2 = Layer_Dense(64, 64)

# Create Sigmoid activation:
activation2 = ReLU_Activation()

# Create third dense layer with 64 input features and 1 output values
layer3 = Layer_Dense(64, 1)

# Create Linear activation:
activation3 = Activation_Linear()

# Create loss function
loss_function = Mean_Squared_Error_Loss()

# Create optimizer object
# (There are parameters that might return better results, so set decay in anyway you want and have fun experimenting!)
# optimizer = Optimizer_SGD(decay=1e-3, momentum=0.9)
# optimizer = Optimizer_Adagrad(decay=1e-4)
# optimizer = Optimizer_RMSProp(learning_rate = 0.02, decay = 1e-5, rho = 0.999)
optimizer = Optimizer_Adam(learning_rate=0.005, decay=1e-3)

# Accuracy precision for accuracy calculation
# We'll calculate this precision as a fraction of standard deviation of all the ground truth values
accuracy_precision = np.std(y) / 250

# Train in loop
for epoch in range (10001):
    # Perform a forward pass of our training data through this layer
    layer1.forward(X)

    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(layer1.output)

    # Perform a forward pass through second Dense layer
    # takes outputs of activation function
    # of first layer as inputs
    layer2.forward(activation1.output)

    # Perform a forward pass through activation function
    # takes the output of second dense layer here
    activation2.forward(layer2.output)

    # Perform a forward pass through third Dense layer
    # takes outputs of activation function of second layer as inputs
    layer3.forward(activation2.output)

    # Perform a forward pass through activation function
    # takes the output of third dense layer here
    activation3.forward(layer3.output)

    # Calculate data loss
    data_loss = loss_function.calculate(activation3.output, y)

    # Calculate regularization penalty
    regularization_loss = loss_function.regularization_loss(layer1) + loss_function.regularization_loss(layer2) + loss_function.regularization_loss(layer3)

    # Calculate overall loss
    loss = data_loss + regularization_loss

    # Calculate accuracy from output of activation2 and targets
    # To calculate it we're taking absolute difference between
    # predictions and ground truth values and compare if differences
    # are lower than given precision value
    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)

    if not epoch % 100:
        print("epoch: {}, acc: {:.3f}, loss: {:.3f} (data_loss: {:.3f}, reg_loss: {:.3f}), lr: {}".format(epoch, accuracy, loss, data_loss, regularization_loss, optimizer.current_learning_rate))

    # Backward pass (Backpropagation)
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.deltaInputs)
    layer3.backward(activation3.deltaInputs)
    activation2.backward(layer3.deltaInputs)
    layer2.backward(activation2.deltaInputs)
    activation1.backward(layer2.deltaInputs)
    layer1.backward(activation1.deltaInputs)

    # Update the weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)
    optimizer.update_params(layer3)
    optimizer.post_update_params()

X_test, y_test = sine_data()

layer1.forward(X_test)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)
layer3.forward(activation2.output)
activation3.forward(layer3.output)

plt.plot(X_test, y_test)
plt.plot(X_test, activation3.output)
plt.show()