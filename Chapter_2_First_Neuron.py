import os

inputs = [1, 2, 3, 2.5]

"""
Each input will also have weights. The values for weights and biases are what get "trained," and they are what make
a model actually work (or not work)
"""
weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]

"""
Since we are modeling a single neuron, we only have one bias, as there is just one bias value per neuron.
The bias is an additional tunable value but is not associated with any input in contrast to the
weights.
"""
bias = [2, 3, 0.5]

"""
A layer is a group of neuron. The method of calculating the output is similar to finding the dot product of a vector.
Since calcuating each neuron manually is too much to handle, we use loops.

The zip() function takes iterables (can be zero or more), aggregates them in a tuple (or a pair), and returns it
"""
layer_outputs = []
# For each neuron
for neuron_weights, neuron_bias in zip(weights, bias):
    neuron_output = 0
    # For each input and weight to the neuron
    for n_input, weight in zip(inputs, neuron_weights):
        # Multiply the input with the associated weight and add it into the output
        neuron_output += n_input * weight
    #Add the output with the bias
    neuron_output += neuron_bias
    #Append to the layer
    layer_outputs.append(neuron_output)

"""
We know we have three neurons because there are 3 sets of weights and 3 biases
"""

os.system("clear")
print(layer_outputs)
