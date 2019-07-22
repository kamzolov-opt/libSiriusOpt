import numpy as np
from .mig import mig


def magic_sag(steps, x0, gradi, func, func_counts, L, A_shape_0):
    gradients, func_res_values = [], []
    xk = x0.copy()
    for j in [np.random.randint(0, func_counts) for _ in range(A_shape_0)]:
        gradients.append(gradi(xk, j))

    h = 0.125 / L
    s = sum(gradients)
    for k in range(steps):
        j = np.random.randint(0, A_shape_0 - 1)
        s -= gradients[j]
        del gradients[j]
        yj = gradi(xk, j)
        gradients.append(yj)
        xk = xk - h * (s / len(s))
        s += yj
        func_res_values.append(func(xk))
    return xk, func_res_values


def magic_sag2(steps, x0, grad, gradi, func, func_counts, L, size, A_learn):
    iter_msag = round(steps * 0.8)
    iter_mig = round(steps * 0.2) // 300
    xk, points = magic_sag(iter_msag, x0, gradi, func, func_counts, L, A_learn)
    xk, points2 = mig(xk, func, grad, gradi, iter_mig, L, size)
    return xk, points + points2


def sag(n, x0, gradi, func, l, steps):
    xk = x0.copy()
    y = [gradi(xk, i) for i in range(n)]
    sum_y = sum(y)
    hk = 1 / (16 * l)
    res = [func(xk)]
    for _ in range(steps):
        j = np.random.randint(1, n)
        sum_y -= y[j]
        y[j] = gradi(xk, j)
        sum_y += y[j]
        xk = xk - hk * (sum_y / n)
        res.append(func(xk))
    return xk, res


