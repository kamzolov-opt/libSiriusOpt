import numpy as np

def Adagrad(xk, grad, func, steps, a=0.9, hk=0.001):
    n = xk.shape[0]
    res = [func(xk)]
    vt = 0
    for i in range(1, steps + 1):
        vt = vt + grad(xk) * grad(xk)
        At = np.eye(n) * vt ** 0.5 + hk * np.eye(n)
        xk = xk - a * np.linalg.inv(At) @ grad(xk)
        res.append(func(xk))
    return xk, res
