#!/usr/bin/env python3

import sys, os, time, math, pickle

dirScript = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirScript)
sys.path.append(os.path.join(dirScript, "python-mnist"))

from mnist import MNIST
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

class DataStorage:
    pass

def serialize(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

def deserialize(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

# Loading test data
def loadData(cacheIsOn = True):
    cacheFileName = "input_cache.bin"

    if cacheIsOn and os.path.exists(cacheFileName):
        print(" !! load data from cache: ", cacheFileName)
        res = deserialize(cacheFileName)
        return res

    absPathToData = os.path.join(dirScript, 'python-mnist/dir_with_mnist_data_files')

    mndata = MNIST(absPathToData)
    mndata.gz = True
    images, labels = mndata.load_training()
  
    res = DataStorage()
    res.images = images
    res.labels = labels
    res.pathToData  = absPathToData
    res.imageWidth  = 28
    res.imageHeight = 28

    n = len(res.images)
    m = res.imageWidth * res.imageHeight + 1

    res.imagesMat = np.ones((n, m))
    res.labelsMat = np.ones((n, 1))
    
    for i in range(n):
        for j in range(res.imageWidth * res.imageHeight):
            res.imagesMat[i, j] = res.images[i][j]
        res.labelsMat[i, 0] = res.labels[i]

    if cacheIsOn:
        print(" !! serialize to cache: ", cacheFileName)
        serialize(res, cacheFileName)

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

# Show image with secific index from numpy arrays
def showImageMat(storage, index):
    print("Show image with index = ", index, ", image label is ", storage.labelsMat[index,0])   
    x = np.asarray(storage.imagesMat[index])[0:-1]
    x = x.reshape((storage.imageWidth, storage.imageHeight))
    image = x
    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    storage = loadData()
    printInfo(storage)
    showImage(storage, 5)

