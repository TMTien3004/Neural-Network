"""
The Dropout function works by randomly disabling neurons at a given rate during every forward pass, forcing the network to 
learn how to make accurate predictions with only a random part of neurons remaining.
"""

# Forward Pass
#---------------------------------------------------------------------------------------------------------------------------
"""
In the code, we will "turn off" neurons with a filter that is an array with the same shape as the layer output but filled 
with numbers drawn from a Bernoulli distribution. A Bernoulli distribution is a binary (or discrete) probability distribution 
where we can get a value of 1 with a probability of p and value of 0 with a probability of q (q = 1 - p).
"""

import numpy as np

dropout_rate = 0.2
# Example output containing 10 values
example_output = np.array([0.27, -1.03, 0.67, 0.99, 0.05, -0.37, -2.01, 1.13, -0.07, 0.73])
print("Sum initial: {}".format(sum(example_output)))

"""
np.random.binomial(n, p, size) will have three input:
- n: number of experiments (or rather how many tosses of the coin do you want to do)
- p: the probability of a toss
- size: size is how many of these `tests` to run, and the return is a list of overall results

np.random.binomial(2, 0.5, size = 10)
=> [0, 0, 1, 2, 0, 2, 0, 1, 0, 2]
"""

sums = []
for i in range(10000):
    # Basically, the point of this is to disable certain neurons. If we put the dropout_rate of 0.3 (30%), then it only "disables"
    # any 3 out of 10 neurons and keep the remaining 7 neurons.
    example_output2 = example_output * np.random.binomial(1, 1 - dropout_rate, example_output.shape) / (1-dropout_rate)
    sums.append(sum(example_output2))

print("Mean sum: {}".format(np.mean(sums)))
