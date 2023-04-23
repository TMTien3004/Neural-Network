import numpy as np

#-------------------------------------------------------------------------------------------------------------------------
# Use this as an alternative to nnfs package
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
X, y = spiral_data(100, 3)
#-------------------------------------------------------------------------------------------------------------------------


#Rectified Linear Activation Function
#-------------------------------------------------------------------------------------------------------------------------

class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # We'll stick with random initialization for now
        self.biases = np.zeros((1, n_neurons))
        pass # using pass statement as a placeholder

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
        pass # using pass statement as a placeholder

class ReLU_Activation:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

#Create a layer
layer1 = Layer_Dense(2,5)
activation1 = ReLU_Activation()

layer1.forward(X)

#print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)