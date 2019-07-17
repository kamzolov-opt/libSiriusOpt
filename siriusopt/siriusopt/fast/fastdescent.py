import numpy as np
from siriusopt.gd import gradient_step


def fast_descent(x0, func, grad, L, steps):
    xk = x0
    uk = x0
    ak = 1 / L
    a_bigk = ak
    res = [func(xk)]
    for _ in range(steps):
        ak = 1 / L * 0.5 + (1 / L ** 2 * 0.25 + ak ** 2) ** 0.5
        yk = (ak * uk + a_bigk * xk) / (a_bigk + ak)
        uk = uk - ak * grad(yk)
        xk = (ak * uk + a_bigk * xk) / (a_bigk + ak)
        a_bigk += ak
        res.append(func(xk))
    return xk, res


def grad_fast_step(grad, func, steps, x, L=0):
    L_new = (1 + np.sqrt(1 + 4 * L * L)) * 0.5
    gamma = (1 - L) / L_new
    x_next = x.copy()
    res = [func(x)]
    for i in range(steps):
        grad_step = gradient_step(x, grad, i)
        x = (1 - gamma) * grad_step + gamma * x_next
        x_next = grad_step

        L = L_new
        L_new = (1 + np.sqrt(1 + 4 * L * L)) * 0.5
        gamma = (1 - L) / L_new
        res.append(func(x_next))
    return x_next, res


def Fast_grad_Nest(x0, func, grad, L, steps):
    xk = x0.copy()
    res = [func(xk)]
    alf = 0
    yk = xk
    for i in range(steps):
        alf_prev = alf
        alf = (1 + np.sqrt(1 + 4 * alf**2)) / 2
        tet = (1 - alf_prev) / alf
        yk_prev = yk
        yk = xk - 1 / L * grad(xk)
        xk = (1 - tet) * yk + tet * yk_prev
        res.append(func(xk))
    return xk, res
