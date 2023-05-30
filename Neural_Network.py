import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
"""
This program MUST be run in Python 3+, so type in the command line: python3 Neural_Network.py
"""

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
    def forward(self, inputs, training):
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
    def forward(self, inputs, training):
        # Save input values 
        self.inputs = inputs
        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return
        # Generate and save scale mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, deltaValues):
        # Gradient on values
        self.deltaInputs = deltaValues * self.binary_mask


# Input "layer"
class Layer_Input:
    # Forward pass
    def forward(self, inputs, training):
        self.output = inputs


# ReLU Activation
class ReLU_Activation:
    # Forward pass
    def forward(self, inputs, training):
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
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
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
    def forward(self, inputs, training):
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
    def forward(self, inputs, training):
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
    def regularization_loss(self):
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
    def calculate(self, output, y, *, include_regularization=False): # y is the intended target values
        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len (sample_losses)

        # If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Calculates accumulated loss
    def calculate_accumulated(self, *, include_regularization=False): # * means that the argument is required
        
        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        # If just data loss - return it
        if not include_regularization:
            return data_loss
        
        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


# Cross-Entropy Loss
class Categorical_Cross_Entropy_Loss(Loss):
    #Forward pass
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

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        # Calculate gradient
        self.deltaInputs = -y_true / deltaValues
        # Normalize gradient
        self.deltaInputs = self.deltaInputs / samples


# Softmax classifier - combined Softmax activation and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():        
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


# Common accuracy class
class Accuracy:
    # Calculates an accuracy
    # given predictions and ground truth values
    def calculate(self, predictions, y):
        # Get comparison results
        comparisons = self.compare(predictions, y)

        # Calculate an accuracy
        accuracy = np.mean(comparisons)

        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        # Return accuracy
        return accuracy

    # Calculates accumulated accuracy
    def calculate_accumulated(self): 
        
        # Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count

        # Return the data and regularization losses
        return accuracy

    # Reset variables for accumulated accuracy
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


# Accuracy calculation for classification model
class Accuracy_Categorical(Accuracy):
    # No initialization is needed
    def init(self, y):
        pass

    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis = 1)
        return predictions == y


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


# Model class
class Model:
    def __init__(self):
        # Create a list of network objects
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None
    
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
        self.input_layer = Layer_Input()
    
        # Count all the objects
        layer_count = len(self.layers)

        # Initialize a list containing trainable layers:
        self.trainable_layers = []
        
        # Iterate the objects
        for i in range(layer_count):

            # If it's the first layer, the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
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
        
        # If output activation is Softmax and
        # loss function is Categorical Cross-Entropy
        # create an object of combined activation
        # and loss function containing
        # faster gradient calculation
        if isinstance(self.layers[-1], Softmax_Activation) and isinstance(self.loss, Categorical_Cross_Entropy_Loss):
            # Create an object of combined activation
            # and loss functions
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()
                            
    # Train the model
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        # Initialize accuracy object
        self.accuracy.init(y)

        # Default value if batch size is not set
        train_steps = 1

        # If there is the validation data
        if validation_data is not None:
            # Set default number of steps for validation as well
            validation_steps = 1

            # For better readability
            X_val, y_val = validation_data

        # Calculate number of steps
        if batch_size is not None:
            train_steps = len (X) // batch_size
            
            # Dividing rounds down. If there are some remaining data but not a full batch, this won't include it. 
            # (We could add an extra step to include it if we wanted to)
            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None :
                validation_steps = len (X_val) // batch_size
                # Dividing rounds down. If there are some remaining data but not a full batch, this won't include it. 
                # (We could add an extra step to include it if we wanted to)
                if validation_steps * batch_size < len (X_val):
                    validation_steps += 1

        # Main training loop
        for epoch in range (1, epochs+1):
            # Print epoch number
            print (f'epoch:{epoch}')

            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over steps
            for step in range(train_steps):

                # If batch size is not set - train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                
                # Otherwise slice a batch
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]
                
                # Perform the forward pass
                output = self.forward(batch_X, training=True)

                # Calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # Perform backward pass
                self.backward(output, batch_y)

                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # Print a summary
                if not step % print_every or step == train_steps - 1:
                    print("step: {}, acc: {:.3f}, loss: {:.3f} (data_loss: {:.3f}, reg_loss: {:.3f}), lr: {}".format(step, accuracy, loss, data_loss, regularization_loss, self.optimizer.current_learning_rate))

            # Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            print("Training, acc: {:.3f}, loss: {:.3f} (data_loss: {:.3f}, reg_loss: {:.3f}), lr: {}".format(epoch_accuracy, epoch_loss, epoch_data_loss, epoch_regularization_loss, self.optimizer.current_learning_rate))

            # If there is the validation data
            if validation_data is not None:
                # Reset accumulated values in loss
                # and accuracy objects
                self.loss.new_pass()
                self.accuracy.new_pass()

                # Iterate over steps
                for step in range(validation_steps):
                    # If batch size is not set -
                    # train using one step and full dataset
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val
                    
                    # Otherwise slice a batch
                    else:
                        batch_X = X_val[step*batch_size:(step+1)*batch_size]
                        batch_y = y_val[step*batch_size:(step+1)*batch_size]
                    
                    # Perform the forward pass
                    output = self.forward(batch_X, training=False)

                    # Calculate the loss
                    self.loss.calculate(output, batch_y)

                    # Get predictions and calculate an accuracy
                    predictions = self.output_layer_activation.predictions(output)
                    self.accuracy.calculate(predictions, batch_y)

                # Get and print validation loss and accuracy
                validation_loss = self.loss.calculate_accumulated()
                validation_accuracy = self.accuracy.calculate_accumulated()

                # Print a summary
                print("Validation, acc: {:.3f}, loss: {:.3f}".format(validation_accuracy, validation_loss))

    # Forward pass
    def forward(self, X, training):
        # Call forward method on the input layer
        self.input_layer.forward(X, training)

        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        
        # Return output
        return layer.output
    
    # Backward pass
    def backward(self, output, y):
        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method on the combined activation/loss
            # this will set dinputs property
            self.softmax_classifier_output.backward(output, y)

            # Since we'll not call backward method of the last layer
            # which is Softmax activation (since we used combined activation/loss object)
            # we need to set dinputs in this object
            self.layers[-1].deltaInputs = self.softmax_classifier_output.deltaInputs

            # Call backward method going through all the objects
            # in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.deltaInputs)
            
            return

            # First call backward method on the loss
            # this will set dinputs property
            self.loss.backward(output, y)

            # Call backward method going through all the objects
            # in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers):
                layer.backward(layer.next.deltaInputs)
                

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

# Loads a MNIST dataset
def load_mnist_dataset (dataset, path):
    
    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []

    # Load the images and labels
    for label in labels:
        for file in os.listdir(os.path.join('fashion_mnist_images', 'train', label)):
            # Read the image
            image = cv2.imread(os.path.join('fashion_mnist_images', 'train', label, file), cv2.IMREAD_UNCHANGED)
            # Append the image and the label to the lists
            X.append(image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')

# MNIST dataset (train + test)
def create_data_mnist(path):
    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    # And return all the data
    return X, y, X_test, y_test

# Create dataset (Load the data)
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[ 0 ], - 1 ).astype(np.float32) - 127.5) / 127.5

# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(X.shape[1], 128))
model.add(ReLU_Activation())
model.add(Layer_Dense(128, 128))
model.add(ReLU_Activation())
model.add(Layer_Dense(128, 10))
model.add(Softmax_Activation())

# Set loss, optimizer and accuracy objects
model.set(loss=Categorical_Cross_Entropy_Loss(), optimizer = Optimizer_Adam(decay=1e-4), accuracy=Accuracy_Categorical())

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data = (X_test, y_test), epochs=10, batch_size=128, print_every=100)