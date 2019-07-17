import numpy as np


def Newton(x0, func, grad, grad2, steps, L):
    xk = x0.copy()
    res = [func(xk)]
    hk = 1 / L
    for i in range(steps):
        xk -= hk * np.linalg.inv(grad2(xk)) @ grad(xk)
        res.append(func(xk))
    return xk, res
