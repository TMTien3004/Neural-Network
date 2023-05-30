# Data Preprocessing
#---------------------------------------------------------------------------------------------------------------------------
# If you want to run this program in Python 3.6, type in the command line: python3 Data_Preprocessing.py

"""
Next, we will scale the data (not the images, but the data representing them, the numbers). Neural networks tend to work 
best with data in the range of either 0 to 1 or -1 to 1, but the pixel values in our images are in the range of 0 to 255.

In this example, we could scale images to be between the range of -1 and 1 by taking each pixel value, subtracting half the 
maximum of all pixel values (i.e., 255/2 = 127.5), then dividing by this same half to produce a range bounded by -1 and 1.
"""
import os
import cv2
import numpy as np

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

# Create dataset (Load the data)
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# Scale features
X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5


"""
Since our Dense layers work on batches of 1-dimensional vectors, They cannot operate on images shaped as a 28x28, 
2-dimensional array. We need to take these 28x28 images and flatten them into 1-dimensional vectors with 784 elements.
"""
# print(X.min(), X.max())
# print(X.shape)

# Reshape to vectors
X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
