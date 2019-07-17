import os, sys

dirScript = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirScript)
sys.path.append(os.path.join(dirScript, './../input_data_read'))

from sklearn.model_selection import train_test_split, KFold
import numpy as np
import math
import input_data_read


class ActivationFuncs:
    SIGMOID = lambda x: 1.0 / (1.0 + np.exp(-x))


class Derivatives:
    def isPrime(n):
        if n == 1 or n == 2 or n == 3:
            return True

        nSqrt = int(math.sqrt(float(n)))
        for i in range(2, nSqrt):
            if (n % i == 0):
                return False
        return True

    def generateRandomIndicies(inputLength, numbersToObtain):
        arr = np.arange(inputLength)
        np.random.shuffle(arr)
        return list(arr[0:numbersToObtain])

    def getBatchRandom(storage, batchSize):
        totalDataMatrixLen = len(storage.images)
        return generateRandomIndicies(totalDataMatrixLen, batchSize)

    def getSequentialBatchCount(storage, batchSize):
        totalDataMatrixLen = len(storage.images)
        return int(totalDataMatrixLen / batchSize)

    def getBatchSequential(storage, batchIndex, batchSize):
        totalDataMatrixLen = len(storage.images)
        iStart = batchIndex * bathSize
        iEnd = (batchIndex + 1) * bathSize
        if (iEnd >= totalDataMatrixLen):
            iEnd = totalDataMatrixLen
        return list(range(iStart, iEnd))

    def crossValidationIndicies(storage, test_size=0.33, shuffle=False):
        totalDataMatrixLen = len(storage.images)

        testLength = int(test_size * totalDataMatrixLen)
        trainLength = int(totalDataMatrixLen - testLength)

        indicies = None
        if shuffle == False:
            indicies = list(range(0, totalDataMatrixLen))
        else:
            indicies = generateRandomIndicies(totalDataMatrixLen, totalDataMatrixLen)

        train_ind = indicies[0:trainLength]
        test_ind = indicies[trainLength:]
        return train_ind, test_ind

    def kFoldCrossValidationIndicies(storage, kFolds, shuffle=False):
        totalDataMatrixLen = len(storage.images)

        indicies = None
        if shuffle == False:
            indicies = list(range(0, totalDataMatrixLen))
        else:
            indicies = generateRandomIndicies(totalDataMatrixLen, totalDataMatrixLen)

        testSizeForFold = int(totalDataMatrixLen / kFolds)

        kFoldsInd = []

        for i in range(kFolds):
            iTestStart = i * testSizeForFold
            iTestEnd = (i + 1) * testSizeForFold

            if (iTestEnd >= totalDataMatrixLen):
                iTestEnd = totalDataMatrixLen

            test_ind = list(indicies[iTestStart:iTestEnd])
            train_ind = list(indicies[0:iTestStart]) + list(indicies[iTestEnd:])
            kFoldsInd.append((train_ind, test_ind))

        return kFoldsInd
