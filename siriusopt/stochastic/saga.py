import numpy as np


def saga(x0, gradi, func, l, steps):
    xk = x0.copy()
    n = xk.shape[0]
    y = [gradi(xk, i) for i in range(n)]
    sum_y = sum(y)
    hk = 1 / (3 * l)
    res = [func(xk)]
    for _ in range(steps):
        j = np.random.randint(1, n)
        sum_y -= y[j]
        gr = gradi(xk, j)
        sum_y += gr
        xk = xk - hk * (gr - y[j] + sum_y / n)
        res.append(func(xk))
        y[j] = gr
    return xk, res


def SAGA_RMS(x0, gradi, func, itern, n, m):
    xk = x0.copy()
    y = [gradi(xk, i) for i in range(n)]
    sum_y = sum(y)
    v = np.zeros((n, m))
    res = [func(xk)]

    for idx in range(1, itern + 1):
        j = np.random.randint(1, n)

        sum_y -= y[j]
        gr = gradi(xk, j)
        sum_y += gr

        # get step
        hk = GetStep(v[j], idx, gr)

        xk = xk - hk @ (gr - y[j] + sum_y / n)
        res.append(func(xk))
        y[j] = gradi(xk, j)
    return xk, res


def GetStep(v, idx, gr, alpha=0.003, beta=0.9, delta=1):
    v = beta * v + (1 - beta) * (gr * gr)
    eps = delta / np.sqrt(idx)
    alpha_new = alpha / np.sqrt(idx)
    A = np.diag(np.sqrt(v)) + eps * np.eye(len(v))
    hk = alpha_new * np.linalg.inv(A)
    return hk