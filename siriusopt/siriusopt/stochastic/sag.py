import numpy as np
from .mig import mig


def sag(steps, x0, grad, func, func_counts, L, A_shape_0):
    gradients, func_res_values = [], []
    xk = x0
    for j in [np.random.randint(0, func_counts) for _ in range(A_shape_0)]:
        gradients.append(grad(xk, j))

    h = 0.125 / L
    s = sum(gradients)
    for k in range(steps):
        j = np.random.randint(0, A_shape_0 - 1)
        s -= gradients[j]
        del gradients[j]
        yj = grad(xk, j)
        gradients.append(yj)
        xk = xk - h * (s / len(s))
        s += yj
        func_res_values.append(func(xk))
    return xk, func_res_values


def magic_sag2(steps, x0, grad, gradi, func, func_counts, L, size, A_learn):
    iter_msag = round(steps * 0.8)
    iter_mig = round(steps * 0.2) // 300
    xk, points = sag(iter_msag, x0, gradi, func, func_counts, L, A_learn.shape[0])
    xk, points2 = mig(xk, func, grad, gradi, iter_mig, L, size)
    return xk, points + points2


def sag_yana(n, x0, grad, func, l, steps):
    xk = x0
    y = [grad(xk, i) for i in range(n)]
    sum_y = sum(y)
    hk = 1 / (16 * l)
    res = [func(xk)]
    for _ in range(steps):
        j = np.random.randint(1, n)
        sum_y -= y[j]
        y[j] = grad(xk, j)
        sum_y += y[j]
        xk = xk - hk * (sum_y / n)
        res.append(func(xk))
    return xk, res


