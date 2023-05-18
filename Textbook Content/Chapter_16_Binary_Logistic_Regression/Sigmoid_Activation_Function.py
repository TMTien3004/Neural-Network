# Sigmoid Activation Function
#---------------------------------------------------------------------------------------------------------------------------
"""
We are now going to cover an alternate output layer option, where each neuron separately represents two classes:
0 for one of the classes, and 1 for the other. A model with this type of output layer is called binary logistic regression.
Binary logistic regression is a regressor type of algorithm, which will differ as we will use a sigmoid activation function
for the output layer rather than softmax.
"""
class Activation_Sigmoid:
    def forward(self, inputs):
        # Save input and calculate/save output
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, deltaValues):
        # Derivative - calculates from output of the sigmoid function
        self.deltaInputs = deltaValues * (1 - self.output) * self.output
