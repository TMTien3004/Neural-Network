import os
#Dot Product & Vector Addition
#-------------------------------------------------------------------------------------------------------------------------
dot_product = 0
vectorA = [1, 2, 3] # What if vectorA represented as inputs?
vectorB = [2, 3, 4] # What if vectorB represented as weights?

for i in range(len(vectorA)):
    dot_product += vectorA[i] * vectorB[i]

os.system("clear")
print(dot_product)

    