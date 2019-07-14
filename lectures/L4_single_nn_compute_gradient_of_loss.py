#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ??????
import numpy as np

# ??????
def makeRandom(npVec): 
    return np.random.random(npVec.shape)

def S0(x): 
    return 1.0

def S(x): 
    return 1.0/(1.0 + np.exp(-x))

# ??????
def dS_dOut(out): 
    return out*(1-out)

# ???? 
class ForwardResult: 
    pass

# ??? 
def forwardEvalute(x_with_intercept, a, b):
    z = np.dot(a.T, x_with_intercept)
    o = np.zeros(shape = z.shape)
    for i in range(len(z)):
        if i == 0: o[i] = S0(z[i])
        else:      o[i] = S(z[i])
    Fhat = np.dot(o,b)

    res = ForwardResult()
    res.x_with_intercept = x_with_intercept
    res.a = a
    res.b = b
    res.z = z
    res.o = o
    res.Fhat = Fhat

    return res

# ???
def partial_DL_db(fwd, y):
    dL_db = (fwd.Fhat - y[0])*fwd.o
    return dL_db

# ???
def partial_DL_da(fwd, y):
    (N,M) = fwd.a.shape
    dL_da = np.zeros((N,M))
    for j in range(N):
        for m in range(M):
            dL_da[j,m] = (fwd.Fhat - y[0]) * fwd.b[m] * (dS_dOut(fwd.o[m])) * fwd.x_with_intercept[j]
    return dL_da

# ???
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
        f_db_plus = forwardEvalute(x_with_intercept, a, b + db)
        f_db_plus_eval = np.square(f_db_plus.Fhat - y[0]) / 2.0

        f_db_minus = forwardEvalute(x_with_intercept, a, b - db)
        f_db_minus_eval = np.square(f_db_minus.Fhat - y[0]) / 2.0

        dL_db[i] = (f_db_plus_eval - f_db_minus_eval) / (2*eps);
    return dL_db

# ???
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
            f_da_plus = forwardEvalute(x_with_intercept, a + da, b)
            f_da_plus_eval = np.square(f_da_plus.Fhat - y[0]) / 2.0

            f_da_minus = forwardEvalute(x_with_intercept, a - da, b)
            f_da_minus_eval = np.square(f_da_minus.Fhat - y[0]) / 2.0

            dL_da[j,m] = (f_da_plus_eval - f_da_minus_eval) / (2*eps);
    return dL_da

#==================================================================================

n = 4 # number of what ???
m = 3 # number of what ???

print("Experiment on (%i,%i,%i) neural network (???,???,output)" % (n,m,1))

y = np.zeros(1)
x = np.zeros(n)

# What does it mean ???
n = n + 1        
m = m + 1     

a = np.zeros((n,m))
b = np.ones(m)

verbose = 0              # ???
experiments = 100        # ???
failedExperiments = 0    # ???

for i in range(experiments):
    y = makeRandom(y) # make random ???
    x = makeRandom(x) # make random ???
    a = makeRandom(a) # Values of a[i,j] where a[i,j] is the weight of the edge connected i-th input unit with j-th hidden unit
    b = makeRandom(b) # Where b[i] connects i-th hidden unit with output unit
    x_with_intercept = np.concatenate((x, np.array([1.0])))

    if verbose:
        print("Y=", y)
        print("X=", x_with_intercept)
        print("A=", a)
        print("b=", b)

    # Forward step
    fwd = forwardEvalute(x_with_intercept, a, b)

    if verbose:
        print("Fhat after forward step: ", fwd.Fhat)
        print("???, i.e. DL_db: ", partial_DL_db(fwd, y))
        print("???, i.e. DL_da: ", np.reshape(partial_DL_da(fwd, y), (1,n*m)))


    predicateCheck1 = np.linalg.norm(partial_DL_db(fwd, y) - partial_DL_db_compute(fwd, y)) < 0.0001
    print("???, i.e. DL_db as numerical: ", predicateCheck1)
    if not predicateCheck1: failedExperiments = failedExperiments + 1

    predicateCheck2 = np.linalg.norm( np.reshape(partial_DL_da(fwd, y), (1,n*m)) - np.reshape(partial_DL_da_compute(fwd, y), (1,n*m))) < 0.001
    print("???, i.e. DL_da as numerical: ", predicateCheck2)
    if not predicateCheck2: failedExperiments = failedExperiments + 1

print("=========================================================")
print("Total number of failed experiments: %i (from total %i)" % (failedExperiments, experiments))
print("=========================================================")
