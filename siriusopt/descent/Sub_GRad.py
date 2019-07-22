import numpy as np


def ternary_search_h(a, b, func, xk, pk):
    while b - a > 1e-3:
        c = a + (b - a) / 3
        d = a + (b - a) / 3 * 2
        if func(xk + c * pk) >= func(xk + d * pk):
            a = c
        else:
            b = d
    return a


def Sub_Grad(x0, iters, func, grad, bet_k=0.01):
    xk = x0.copy()
    pk = x0.copy()
    res = [func(xk)]
    prev_gr = grad(xk)
    for i in range(iters):
        hk = ternary_search_h(0.0001, 1000, func, xk, pk)
        xk -= hk * pk
        grad_x = grad(xk)
        pk = grad_x - bet_k * pk
        bet_k = grad_x @ (grad_x - prev_gr) / np.linalg.norm(prev_gr)**2
        prev_gr = grad_x
        res.append(func(xk))
    return xk, res
