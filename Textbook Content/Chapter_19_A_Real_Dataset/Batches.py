# Batches
#---------------------------------------------------------------------------------------------------------------------------
# If you want to run this program in Python 3.6, type in the command line: python3 Batches.py

# Each batch of samples being trained is referred to as a step. We can calculate the number of steps by dividing the 
# number of samples by the batch size:
steps = X.shape[0] // BATCH_SIZE