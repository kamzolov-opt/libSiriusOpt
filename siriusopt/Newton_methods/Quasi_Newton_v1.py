import numpy as np


def Quasi_Newton_v1(x0, func, grad, grad_2, steps, L):
    xk = x0.copy()
    res = [func(xk)]
    hk = 1 / L
    for i in range(steps):
        xk_prev = xk.copy()
        xk -= hk * np.linalg.inv(grad_2(xk)) @ grad(xk)
        delt_grad = grad(xk) - grad(xk_prev)
        delt_x = xk - xk_prev
        hk += (delt_x - hk * delt_grad) @ (delt_x - hk * delt_grad).T /  ((delt_x - hk * delt_grad) @ delt_grad)
        res.append(func(xk))
    return xk, res
