import numpy as np


def Adagrad(xk, T, grad, func, a=0.01, hk=0.001):
    res = [func(xk)]
    n = xk.shape[0]
    vt = 0
    for i in range (1, T + 1):
        vt += grad(xk) * grad(xk)
        At = np.eye (n) * vt**0.5 + hk * np.eye(n)
        xk = xk - a * np.linalg.inv(At) @ grad(xk)
        res.append(func(xk))
    return xk, res