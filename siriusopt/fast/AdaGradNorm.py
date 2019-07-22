import numpy as np


def Adagrad_norm(x0, func, grad_i, step, bj=100, nu=2):
    xk = x0.copy()
    res = [func(xk)]
    n = 28
    for k in range(step):
        ex = np.random.randint(1, 300)
        now_gr = grad_i(xk, ex)
        bj += np.linalg.norm(now_gr)**2
        xk -= nu / bj**0.5 * now_gr
        res.append(func(xk))
    return xk, res
