import numpy as np


def mig(x0, func, grad, gradi, itera, L, size, m=None):
    if m is None:
        m = size
    xk = x0.copy()
    matr_x = x0.copy()
    x_size = np.size(xk)
    matr_x_avg = np.zeros(x_size)
    res = [func(xk)]
    itera //= m
    for k in range(itera):
        now_gr = grad(xk)
        tet = 2 / (k + 5)
        param_n = 1 / (4 * L * tet)
        for i in range(m):
            j = np.random.randint(0, size)
            tmp = tet * matr_x + (1 - tet) * xk
            tmp_grad = gradi(tmp, j) - gradi(xk, j) + now_gr/size
            matr_x -= tmp_grad * param_n
            matr_x_avg = (matr_x_avg * i + matr_x) / (i + 1)
            res.append(func(xk))
        xk = tet * matr_x_avg + (1 - tet) * xk
        res.append(func(xk))
    return xk, res
