import numpy as np


def RMS(x0, grad, func, t, a=2, b=0.9, delta=0.00009, v=0):
    xk = x0.copy()
    res = [func(xk)]
    for idx in range(1, t + 1):
        g = grad(xk)
        v = b * v + (1 - b) * (g * g)

        e = delta / np.sqrt(idx)
        a_new = a / np.sqrt(idx)

        A = np.diag(np.sqrt(v)) + e * np.eye(len(v))
        xk = xk - a_new * np.linalg.inv(A) @ g
        res.append(func(xk))
    return xk, res
