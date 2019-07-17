#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

#===============================================================================

def S0(x):
    return 1.0

def S(x):
    return 1.0 / (1.0 + np.exp(-x))

def dS_dOut(out):
    return out * (1 - out)

#===============================================================================

class ForwardResult:
    pass



#===============================================================================

def forwardEvalute(x_with_intercept, a, b, cfg = None):
    z = np.dot(a.T, x_with_intercept)
    o = np.zeros(shape=z.shape)
    for i in range(len(z)):
        if i == 0:
            o[i] = S0(z[i])
        else:
            o[i] = cfg.function(z[i])
    Fhat = np.dot(o.T, b)

    res = ForwardResult()
    res.x_with_intercept = x_with_intercept
    res.a = a
    res.b = b
    res.z = z
    res.o = o
    res.Fhat = Fhat

    return res

def partial_DL_db(fwd, y, cfg = None):
    dL_db = (fwd.Fhat - y[0]) * fwd.o
    return dL_db

def partial_DL_da(fwd, y, cfg = None):
    (N, M) = fwd.a.shape
    dL_da = np.zeros((N, M))
    for j in range(N):
        for m in range(M):
            dL_da[j, m] = (fwd.Fhat - y[0]) * fwd.b[m] * (dS_dOut(fwd.o[m])) * fwd.x_with_intercept[j]
    return dL_da

def partial_DL_db_compute(fwd, y, cfg):
    x_with_intercept = fwd.x_with_intercept
    a = fwd.a
    b = fwd.b
    (n, m) = a.shape
    eps = 0.0001

    dL_db = np.zeros(m)
    for i in range(m):
        db = np.zeros(m)
        db[i] = eps
        f_db_plus = forwardEvalute(x_with_intercept, a, b + db, cfg)
        f_db_plus_eval = np.square(f_db_plus.Fhat - y[0]) / 2.0

        f_db_minus = forwardEvalute(x_with_intercept, a, b - db, cfg)
        f_db_minus_eval = np.square(f_db_minus.Fhat - y[0]) / 2.0

        dL_db[i] = (f_db_plus_eval - f_db_minus_eval) / (2 * eps);
    return dL_db

def partial_DL_da_compute(fwd, y, cfg = None):
    x_with_intercept = fwd.x_with_intercept
    a = fwd.a
    b = fwd.b
    (N, M) = a.shape
    eps = 0.0001

    dL_da = np.zeros(a.shape)
    for j in range(N):
        for m in range(M):
            da = np.zeros((N, M))
            da[j, m] = eps
            f_da_plus = forwardEvalute(x_with_intercept, a + da, b, cfg)
            f_da_plus_eval = np.square(f_da_plus.Fhat - y[0]) / 2.0

            f_da_minus = forwardEvalute(x_with_intercept, a - da, b, cfg)
            f_da_minus_eval = np.square(f_da_minus.Fhat - y[0]) / 2.0

            dL_da[j, m] = (f_da_plus_eval - f_da_minus_eval) / (2 * eps);
    return dL_da
