# Learning Rate Decay
#---------------------------------------------------------------------------------------------------------------------------
"""
- A commonly-used solution to keep initial updates large and explore various learning rates during training is to implement 
a learning rate decay.

- The idea of a learning rate decay is to start with a large learning rate, say 1.0 in our case, and then decrease it 
during training. The method we will implement here is to program a Decay Rate, which steadily decays the learning rate 
per batch or epoch.

Decay Rate: Basically, we are going to update the learning rate each step by the reciprocal of the step count fraction.
"""
starting_learning_rate = 1.0
learning_rate_decay = 0.1
for step in range(20):
    learning_rate = starting_learning_rate * (1. / (1 + learning_rate_decay * step)) 
    print(learning_rate)