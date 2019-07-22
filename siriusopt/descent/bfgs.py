import numpy as np


def bin_search(a, b, pk, xk, grad, func, c1):
    while b - a > 1e-6:
        c = (a + b) / 2
        if func(xk + c * pk) > func(xk) + c1 * b * grad(xk).T @ pk:
            b = c
        else:
            a = c
    return b


def bin_search_ray_max(a, b, pk, xk, grad, func, c1, c2):
    c = (a + b) / 2
    if abs(grad(xk + c * pk).T @ pk) > c2 * abs(grad(xk).T @ pk):
        return bin_search(a, c, pk, xk, grad, func, c1)
    else:
        return bin_search_ray_max(c, 2 * b, pk, xk, grad, func, c1, c2)


def bfgs(x0, func, grad, steps, c1=1e-3, c2=0.9):
    xk = x0.copy()
    res = [func(xk)]
    hk = np.ones((len(xk), len(xk))) * 0.1
    for i in range(steps):
        xk_prev = xk.copy()
        pk = -hk @ grad(xk)
        alf = bin_search_ray_max(0, 10, pk, xk, grad, func, c1, c2)
        xk += alf * pk

        delt_grad = grad(xk) - grad(xk_prev)
        delt_x = xk - xk_prev

        p_not_pk = 1 / (delt_grad.T @ delt_x)
        t = np.outer(delt_grad, delt_x)
        tm = np.eye(len(xk)) - p_not_pk * t
        tmp = np.eye(len(xk)) - p_not_pk * t.T
        hk = tmp @ hk @ tm + p_not_pk * np.outer(delt_x, delt_x)
        res.append(func(xk))
    return xk, res
