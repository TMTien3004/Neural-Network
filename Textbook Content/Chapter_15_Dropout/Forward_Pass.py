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

import random
dropout_rate = 0.5
# Example output containing 10 values
example_output = [ 0.27 , - 1.03 , 0.67 , 0.99 , 0.05 , - 0.37 , - 2.01 , 1.13 , - 0.07 , 0.73 ]

while True :
# Randomly choose index and set value to 0
    index = random.randint( 0 , len (example_output) - 1 )
    example_output[index] = 0
    # We might set an index that already is zeroed
    # There are different ways of overcoming this problem,
    # for simplicity we count values that are exactly 0
    # while it's extremely rare in real model that weights
    # are exactly 0, this is not the best method for sure
    dropped_out = 0
    for value in example_output:
        if value == 0 :
            dropped_out += 1

    # If required number of outputs is zeroed - leave the loop
        if dropped_out / len (example_output) >= dropout_rate:
            break

print(example_output)
print("Hello?")