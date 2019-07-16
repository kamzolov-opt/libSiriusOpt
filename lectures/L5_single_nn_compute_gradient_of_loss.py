#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np

def makeRandom(npVec): 
    return np.random.random(npVec.shape)


# Below
# -- z is intermediate variable which is unput to activation unit
# -- out is intermediate variable which is equal to out = S(z)
def S0(z): 
    return 1.0
def dS0_dz(out, z): 
    return 0.0

def sigmoid(z): 
    return 1.0/(1.0 + np.exp(-z))
def dsigmoid_dz(out, z):
    return out*(1-out)

# Dummy forward result class
class ForwardResult: 
    pass

# Evaluate forward step
def forwardEvalute(x_with_intercept, a, b, actFuncs):
    z = np.dot(x_with_intercept, a)
    o = np.zeros(shape = z.shape)

    for i in range(0, o.shape[0]):
        for j in range(0, o.shape[1]):
            o[i,j] = actFuncs[j] (z[i,j])

    Fhat = np.dot(o,b)

    res = ForwardResult()
    res.x_with_intercept = x_with_intercept
    res.a = a
    res.b = b
    res.z = z
    res.o = o
    res.Fhat = Fhat
    res.actFuncs = actFuncs

    return res

# Analytic partial derivatives of the quadratic loss w.r.t. to b
def partial_DL_db(fwd, y):
    print((fwd.Fhat - y).shape)
    print((fwd.o).shape)

    sys.exit(-1)

       
    dL_db = (fwd.Fhat - y)*fwd.o
    return dL_db


# Analytic partial derivatives of the quadratic loss w.r.t. to a
def partial_DL_da(fwd, y, actFuncsDerivative):
    (N,M) = fwd.a.shape
    dL_da = np.zeros((N,M))
    for j in range(N):
        for m in range(M):
            dL_da[j,m] = (fwd.Fhat - y[0]) * fwd.b[m] * (actFuncsDerivative[m] (fwd.o[m], fwd.z[m]) ) * fwd.x_with_intercept[j]
    return dL_da

# Numerical partial derivatives of the quadratic loss w.r.t. to b
def partial_DL_db_compute(fwd, y):
    x_with_intercept = fwd.x_with_intercept
    a = fwd.a
    b = fwd.b
    (n,m) = a.shape
    eps = 0.0001

    dL_db = np.zeros(m)
    for i in range(m):
        db = np.zeros(m)
        db[i] = eps
        f_db_plus = forwardEvalute(x_with_intercept, a, b + db, fwd.actFuncs)
        f_db_plus_eval = np.square(f_db_plus.Fhat - y[0]) / 2.0

        f_db_minus = forwardEvalute(x_with_intercept, a, b - db, fwd.actFuncs)
        f_db_minus_eval = np.square(f_db_minus.Fhat - y[0]) / 2.0

        dL_db[i] = (f_db_plus_eval - f_db_minus_eval) / (2*eps);
    return dL_db

# Numerical partial derivatives of the quadratic loss w.r.t. to a
def partial_DL_da_compute(fwd, y):
    x_with_intercept = fwd.x_with_intercept
    a = fwd.a
    b = fwd.b
    (N,M) = a.shape
    eps = 0.0001

    dL_da = np.zeros(a.shape)
    for j in range(N):
        for m in range(M):
            da = np.zeros((N,M))
            da[j,m] = eps
            f_da_plus = forwardEvalute(x_with_intercept, a + da, b, fwd.actFuncs)
            f_da_plus_eval = np.square(f_da_plus.Fhat - y[0]) / 2.0

            f_da_minus = forwardEvalute(x_with_intercept, a - da, b, fwd.actFuncs)
            f_da_minus_eval = np.square(f_da_minus.Fhat - y[0]) / 2.0

            dL_da[j,m] = (f_da_plus_eval - f_da_minus_eval) / (2*eps);
    return dL_da

#==================================================================================

n = 4         # number of scalar real inputs
m = 3         # number of hidden units
examples = 10 # examples in train set

print("Experiment on (%i,%i,%i) neural network (inputs,hidden,output)" % (n,m,1))

y = np.zeros((examples, 1)) # target values
x = np.zeros((examples, n)) # input data

n = n + 1  # append intercept term        
m = m + 1  # append fake activation function which value is equal to "1" always   

# Two array one is with activation functions and second with it's derivatives
actFuncs           = [S0, sigmoid, sigmoid, sigmoid]
actFuncsDerivative = [dS0_dz, dsigmoid_dz, dsigmoid_dz, dsigmoid_dz]

a = np.zeros((n,m))
b = np.ones((m,1))

verbose = True
           
# Fill Input data randomly 
y = makeRandom(y) # make random output vector
x = makeRandom(x) # make random input vector

# Initialize weights randomly
a = makeRandom(a) # Values of a[i,j] where a[i,j] is the weight of the edge connected i-th input unit with j-th hidden unit
b = makeRandom(b) # Where b[i] connects i-th hidden unit with output unit

# Extra data preparation
oneColumn = np.ones((examples, 1) )
x_with_intercept = np.hstack((x, oneColumn))

if verbose:
    print("Y=", y)
    print("X=", x_with_intercept)
    print("A=", a)
    print("b=", b)

# Forward step
fwd = forwardEvalute(x_with_intercept, a, b, actFuncs)

if verbose:
    print("Fhat after forward step: ", fwd.Fhat)
    print("Quadratic loss with respect to b, i.e. DL_db: ", partial_DL_db(fwd, y))
    print("Quadratic loss with resect to b, i.e. DL_da: ", np.reshape(partial_DL_da(fwd, y, actFuncsDerivative), (1,n*m)))

    predicateCheck1 = np.linalg.norm(partial_DL_db(fwd, y) - partial_DL_db_compute(fwd, y)) < 0.0001
    print("Quadratic loss with resect to b, i.e. DL_db as numerical: ", predicateCheck1)

    predicateCheck2 = np.linalg.norm( np.reshape(partial_DL_da(fwd, y, actFuncsDerivative), (1,n*m)) - np.reshape(partial_DL_da_compute(fwd, y), (1,n*m))) < 0.001
    print("Quadratic loss with resect to b, i.e. DL_da as numerical: ", predicateCheck2)

print("=========================================================")
