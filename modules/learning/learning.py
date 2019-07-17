#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time
import numpy as np

dirScript = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirScript)
sys.path.append(os.path.join(dirScript, "./../../siriusopt"))
sys.path.append(os.path.join(dirScript, "./../input_data_read"))
sys.path.append(os.path.join(dirScript, "./../sampling"))
sys.path.append(os.path.join(dirScript, "./../single_nn_learning"))

import siriusopt
import input_data_read
import sampling
import single_nn_learning
import actfuncs

if __name__ == "__main__":
    t0 = time.time()  
    data_storage = input_data_read.loadData()
    loadData_time = time.time() - t0
    t0 = time.time()
    print("Time to load data: ", str(loadData_time), " seconds")

    single_nn_learning.X = data_storage.imagesMat
    single_nn_learning.Y = data_storage.labelsMat

#    input_data_read.showImageMat(data_storage, 4)
#    input_data_read.showImage(data_storage, 4)
    single_nn_learning.cfg.function = actfuncs.ActivationFuncs.SIGMOID
    single_nn_learning.cfg.m = 10                                      
    single_nn_learning.cfg.n = single_nn_learning.X.shape[1]      
    single_nn_learning.cfg.totalSamples = 20 #single_nn_learning.X.shape[0]
    single_nn_learning.Indicies = sampling.getBatchSequential(data_storage, 0, single_nn_learning.cfg.totalSamples)
   
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
    optParams, _ = siriusopt.stochastic.sgd(x0 = allParams, grad = single_nn_learning.empiricalRiskGradientWithIndex, steps = 5, func = single_nn_learning.empiricalRisk, L = 1000)
    solve_time = time.time() - t0
    print("Time to solve neural net:  ", str(solve_time), " seconds")

    print("AFTER LEARNING: Current emppirical risk:", single_nn_learning.empiricalRisk(optParams))
    print("AFTER LEARNING: Empirical risk gradient l2 norm: ", np.linalg.norm(single_nn_learning.empiricalRiskGradient(optParams)))
