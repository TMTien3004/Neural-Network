import os

labels = os.listdir('fashion_mnist_images/train')
print(labels)

"""
There are 6000 samples for each class, and 10 classes in total, which means that there are 60000 images in the training set.
"""
files = os.listdir('fashion_mnist_images/train/0')
print(files[:10])
print(len(files))