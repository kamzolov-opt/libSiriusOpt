import numpy as np


def saga(x0, grad, func, l, steps):
    xk = x0
    n = x0.shape[0]
    y = [grad(xk, i) for i in range(n)]
    sum_y = sum(y)
    hk = 1 / (3 * l)
    res = [func(xk)]
    for _ in range(steps):
        j = np.random.randint(1, n)
        sum_y -= y[j]
        gr = grad(xk, j)
        sum_y += gr
        xk = xk - hk * (gr - y[j] + sum_y / n)
        res.append(func(xk))
        y[j] = gr
    return xk, res

