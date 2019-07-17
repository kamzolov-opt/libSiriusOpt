#!/usr/bin/env python3

import sys, os, time, math

dirScript = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirScript)
sys.path.append(os.path.join(dirScript, "python-mnist"))

from mnist import MNIST
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

class DataStorage:
    pass

# Loading test data
def loadData():
    absPathToData = os.path.join(dirScript, 'python-mnist/dir_with_mnist_data_files')

    mndata = MNIST(absPathToData)
    mndata.gz = True
    images, labels = mndata.load_training()
  
    res = DataStorage()
    res.images = images
    res.labels = labels
    res.pathToData = absPathToData
    res.imageWidth = 28
    res.imageHeight = 28
    return res

# Print information about storage
def printInfo(storage):
    print(">> number of images ", len(storage.images))
    print(">> number of labels ", len(storage.labels))
    print(">> path from which data have been obtained from:", storage.pathToData)
    print(">> size of images [%i,%i]" % (storage.imageWidth, storage.imageHeight))

# Show image with secific index
def showImage(storage, index):
    print("Show image with index = ", index, ", image label is ", storage.labels[index])   
    x = np.array(storage.images[index])
    x = x.reshape((storage.imageWidth, storage.imageHeight))
    image = x
    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    storage = loadData()
    printInfo(storage)
    showImage(storage, 5)

