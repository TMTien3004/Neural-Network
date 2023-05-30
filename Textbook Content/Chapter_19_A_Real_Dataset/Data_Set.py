# Data Preparation
#---------------------------------------------------------------------------------------------------------------------------
from zipfile import ZipFile
import os
import urllib
import urllib.request

# If you want to run this in Python 3.6, type in the command line: python3 Data_Set.py
# Make sure that you go to the directory where this file is located before running the command above.

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

if not os.path.isfile(FILE):
    print(f'Downloading {URL} and saving as {FILE} ...')
    urllib.request.urlretrieve(URL, FILE)
    

print('Unzipping images...')
with ZipFile(FILE) as zip_images:
    zip_images.extractall(FOLDER)

print('Done!')


