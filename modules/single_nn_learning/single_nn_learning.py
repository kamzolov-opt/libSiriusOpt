#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, time, math, sys

dirScript = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirScript)

from single_nn import forwardEvalute
from single_nn import partial_DL_da
from single_nn import partial_DL_db
from single_nn import ActivationFuncs

import numpy as np

X = None              # Data Matrix, which stores examples by rows
Y = None              # Label Vector, which stores examples by items
Indicies = None       # Indicies of examples used for empiricalRisk() and 

class Configuration:
   pass

cfg = Configuration() # Configuration of Neural NEt

def makeRandom(x):
    return np.random.random(x.shape)

def packParameterToVector(a, b):
    '''
    a - is a matrix with coefficients for neural net
    b - another coefficients
    '''
    aAsVector = a.reshape((a.size, 1))
    bAsVector = b.reshape((b.size, 1))

    allParameters = np.concatenate( (aAsVector, bAsVector) )
    return allParameters

def unpackParameterFromVector(allParams, cfg):
    '''
    a - is a matrix with coefficients for neural net
    b - another coefficients
    '''
    a = allParams[0:cfg.n * cfg.m, 0]
    b = allParams[cfg.n * cfg.m:, 0]

    a = a.reshape( (cfg.n, cfg.m) )
    b = b.reshape( (cfg.m, 1) )

    return a,b

def getZeroParams():
    params = np.zeros((cfg.n * cfg.m + cfg.m, 1))
    return params

def empiricalRisk(x):
    '''
    x is a parameters of neural net which consist of [a, b]
    step1 - from x extract [a, b]
    step2 - which examples we should take
    step3 - substitute into forwardEvalute(...)
    '''

    a, b = unpackParameterFromVector(x, cfg)
    results = 0.0
    
    for i in range(Indicies.size):
        xi = X[i, :]
        xi = xi.reshape( (xi.size, 1) )
        yi = Y[i]   

        fwd = forwardEvalute(xi, a, b, cfg)
        results += ((fwd.Fhat - yi) ** 2) / 2.0

    return results / Indicies.size

def empiricalRiskGradientWithIndex(x, index):
    '''
    Have no idea for what index correspond to
    '''
    return empiricalRiskGradient(x)


def empiricalRiskGradient(x):
    '''
    Iterate through all indicies Indicies and take x[i] y[i]
    step1 - forwardEvalute
    step2 - np.asarray([partial_DL_da(fwd, y), partial_DL_db(fwd, y)])
    step3 - accumulate final gradient
    '''
    
    a, b = unpackParameterFromVector(x, cfg)
    gradient = None
    
    for i in range(Indicies.size):
        xi = X[i, :]
        xi = xi.reshape( (xi.size, 1) )
        yi = Y[i]   

        fwd = forwardEvalute(xi, a, b, cfg)
        grad_a = partial_DL_da(fwd, yi)
        grad_b = partial_DL_db(fwd, yi)

        grad_to_all_params = packParameterToVector(grad_a, grad_b)

        if gradient is None:
            gradient = grad_to_all_params
        else:
            gradient += grad_to_all_params
 
    return gradient / Indicies.size
 
if __name__ == "__main__":
    #global X, Y

    # X global variable, matrix, store all examples by ROWS
    X = np.array([[1, 2, 3, 1.0],
                  [4, 5, 6, 1.0]
                 ]
                 )

    # Y global variable, column, store all correct values
    Y = np.array([[7],
                  [9]
                 ])

    # Indicies by which pull examples
    Indicies = np.asarray(np.array([[1, 0]]))

    cfg.m = 10                    # number of activation functions
    cfg.n = X.shape[1]            # number of input attributes
    cfg.totalSamples = X.shape[0] # total number of examples

    cfg.function = ActivationFuncs.SIGMOID

    a = np.zeros((cfg.n, cfg.m))
    b = np.ones((cfg.m, 1))

    allParams = packParameterToVector(a, b)
    score = empiricalRisk(allParams)
    print(">> Current emppirical risk:", score)

    grad = empiricalRiskGradient(allParams)
    print(">> Empirical risk gradient l2 norm: ", np.linalg.norm(grad))
