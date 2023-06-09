# Data Loading
#---------------------------------------------------------------------------------------------------------------------------
# If you want to run this program in Python 3.6, type in the command line: python3 Data_Loading.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Loads a MNIST dataset
def load_mnist_dataset (dataset, path):
    
    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []

    # Load the images and labels
    for label in labels:
        for file in os.listdir(os.path.join('fashion_mnist_images', 'train', label)):
            # Read the image
            image = cv2.imread(os.path.join('fashion_mnist_images', 'train', label, file), cv2.IMREAD_UNCHANGED)
            # Append the image and the label to the lists
            X.append(image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')

# MNIST dataset (train + test)
def create_data_mnist(path):
    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    # And return all the data
    return X, y, X_test, y_test

# Create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')
"""
There are 6000 samples for each class, and 10 classes in total, which means that there are 60000 images in the training set.
Since the number of samples for each class is the same, the dataset is balanced.
"""

# # Count the number of images in each class
# files = os.listdir('fashion_mnist_images/train/0')
# print(files[:10])
# print(len(files))

# # Load an image using OpenCV
# image_data = cv2.imread( 'fashion_mnist_images/train/4/0011.png' , cv2.IMREAD_UNCHANGED)
# print(image_data)

# # Display the image on the terminal
# np.set_printoptions(linewidth = 200)

# # Print the image data on the screen
# plt.imshow(image_data, cmap = 'gray')
# plt.show()