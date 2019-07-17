import numpy as np


def SAGAnew(x0, grad, func, itern, n, m):
    xk = x0
    y = [grad(xk, i) for i in range(n)]
    sum_y = sum(y)
    v = np.zeros((n, m))
    res = [func(xk)]

    for idx in range(1, itern):
        j = np.random.randint(1, n)

        sum_y -= y[j]
        gr = grad(xk, j)
        sum_y += gr

        # get step
        hk = GetStep(v[j], idx, gr)

        xk = xk - hk @ (gr - y[j] + sum_y / n)
        res.append(func(xk))
        y[j] = grad(xk, j)
    return xk, res


def RMS(x0, a, b, delta, t, grad, func, v=0):
    xk = x0
    res = [func(xk)]
    for idx in range(1, t):
        g = grad(xk)
        v = b * v + (1 - b) * (g * g)

        e = delta / np.sqrt(idx)
        a_new = a / np.sqrt(idx)

        A = np.diag(np.sqrt(v)) + e * np.eye(len(v))
        xk = xk - a_new * np.linalg.inv(A) @ g
        res.append(func(xk))
    return xk, res


def GetStep(v, idx, gr, alpha=0.003, beta=0.9, delta=1):
    v = beta * v + (1 - beta) * (gr * gr)
    eps = delta / np.sqrt(idx)
    alpha_new = alpha / np.sqrt(idx)
    A = np.diag(np.sqrt(v)) + eps * np.eye(len(v))
    hk = alpha_new * np.linalg.inv(A)
    return hk
