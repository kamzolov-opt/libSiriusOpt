#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time
import numpy as np
import matplotlib.pyplot as plt

dirScript = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirScript)
sys.path.append(os.path.join(dirScript, "./../../siriusopt"))
sys.path.append(os.path.join(dirScript, "./../input_data_read"))
sys.path.append(os.path.join(dirScript, "./../sampling"))
#sys.path.append(os.path.join(dirScript, "./../single_nn_learning"))
#sys.path.append(os.path.join(dirScript, "./../single_nn_learning_batch"))

import siriusopt
import input_data_read
import sampling

import single_nn_learning
import actfuncs

#import single_nn_learning_batch as single_nn_learning
#import actfuncs_batch as actfuncs

if __name__ == "__main__":
    t0 = time.time()  
    data_storage = input_data_read.loadData()
    loadData_time = time.time() - t0
    t0 = time.time()
    print("Time to load data: ", str(loadData_time), " seconds")

    single_nn_learning.X = data_storage.imagesMat[0:10,:]
    single_nn_learning.Y = data_storage.labelsMat[0:10,:]

#    input_data_read.showImageMat(data_storage, 4)
#    input_data_read.showImage(data_storage, 4)
#    print(dir(actfuncs.ActivationFuncs))
    single_nn_learning.cfg.function = actfuncs.ActivationFuncs.TH
    single_nn_learning.cfg.derivative = actfuncs.ActivationFuncs.get_derivative(single_nn_learning.cfg.function)
    single_nn_learning.cfg.m = 10                                      
    single_nn_learning.cfg.n = single_nn_learning.X.shape[1]      
    single_nn_learning.cfg.totalSamples = single_nn_learning.X.shape[0]
    single_nn_learning.cfg.batchSize = 2
    single_nn_learning.updateIndicies(sampling.getBatchSequential(data_storage, 0, single_nn_learning.cfg.batchSize))

    print(" number of activation functions: ", single_nn_learning.cfg.m)
    print(" number of input attributes: ", single_nn_learning.cfg.n)
    print(" number of total examples: ", single_nn_learning.cfg.totalSamples)

    allParams = single_nn_learning.getZeroParams() 
    prepareNN_time = time.time() - t0
    t0 = time.time()
    print("Time to configure neural net: ", str(prepareNN_time), " seconds")

    print("BEFORE LEARNING: Current emppirical risk:", single_nn_learning.empiricalRisk(allParams))
    print("BEFORE LEARNING: Empirical risk gradient l2 norm: ", np.linalg.norm(single_nn_learning.empiricalRiskGradient(allParams)))

    t0 = time.time()

    # Total number of batches  
    totalBatches = int((single_nn_learning.cfg.totalSamples + single_nn_learning.cfg.batchSize - 1) / single_nn_learning.cfg.batchSize)
    maxEpocs = 5
    #totalBatches = 4

    values = []

    for epoc in range(maxEpocs):
        print("epoc #", epoc)

        values.append(single_nn_learning.empiricalRisk(allParams))
        for batch in range(totalBatches):  
            single_nn_learning.updateIndicies(sampling.getBatchSequential(data_storage, batch, single_nn_learning.cfg.batchSize))
            allParams, points = siriusopt.sgd(x0 = allParams, grad = single_nn_learning.empiricalRiskGradientWithIndex, steps = 2, func = single_nn_learning.empiricalRisk, L = 100.0)

    solve_time = time.time() - t0
    print(">>>>>>>>>>>>", values)

    points = [el for el in values]
    #print(optParams)
    print("Series of optimal values: ", points)
    siriusopt.show([points], namefile="sgd", labels=["sgd"])
    print("Time to solve neural net:  ", str(solve_time), " seconds")

    print("AFTER LEARNING: Current emppirical risk:", single_nn_learning.empiricalRisk(allParams))
    print("AFTER LEARNING: Empirical risk gradient l2 norm: ", np.linalg.norm(single_nn_learning.empiricalRiskGradient(allParams)))
