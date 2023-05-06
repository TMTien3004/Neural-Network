# Stochastic Gradient Decent (SGD)
#---------------------------------------------------------------------------------------------------------------------------
"""
Once we have calculated the gradient, we can use this information to adjust weights and biases to decrease the measure of 
loss.

What makes Machine Learning so extraordinary is the implementation of optimization algorithms/formulas, and we can even say
that optimization is the backbone of ML.

There are multiple optimized algorithms:
- Stochastic Gradient Descent: an optimizer that fits a single sample at a time
- Batch Gradient Descent: an optimizer used to fit a whole dataset at once
- Mini-batch Gradient Descent: an optimizer used to fit slices of a dataset.

In the context of the book, we will call slices of data as batches.
In the case of Stochastic Gradient Descent, we choose a learning rate, such as 1.0.

A full pass of a training data is called epoch, where we simply repeatedly perform a forward pass, backward pass, and 
optimization until we reach some stopping point.
"""

# The optimizer’s task is to decrease loss, not raise accuracy directly
class Optimizer_SGD:
    # Initialize the default learning rate of 1.0.
    def __init__(self, learning_rate = 1.0):
        self.learning_rate = learning_rate

    # Update paramaters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.deltaWeights
        layer.biases += -self.learning_rate * layer.deltaBiases

# Create first dense layer with 2 input features and 64 output values
layer1 = Layer_Dense(2, 64)

# Create ReLU Activation
activation1 = ReLU_Activation()

# Create second dense layer with 64 input features and 3 output values
layer2 = Layer_Dense(64, 3)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create optimizer object
optimizer = Optimizer_SGD()

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
    # Print accuracy
    print ("Acc:" , accuracy)

    if not epoch % 100:
        print("epoch: {}, acc: {:.3f}, loss: {:.3f}".format(epoch, accuracy, loss))

    # Backward pass (Backpropagation)
    loss_activation.backward(loss_activation.output, y)
    layer2.backward(loss_activation.deltaInputs)
    activation1.backward(layer2.deltaInputs)
    layer1.backward(activation1.deltaInputs)

    # Update the weights and biases
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)    