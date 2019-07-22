import numpy as np


def ternary_search_h(a, b, func, yk, grad_yk):
    while b - a > 1e-3:
        c = a + (b - a) / 3
        d = a + (b - a) / 3 * 2
        if func(yk - c * grad_yk) >= func(yk - d * grad_yk):
            a = c
        else:
            b = d
    return a


def ternary_search_bet(a, b, xk, vk, func):
    while b - a > 1e-3:
        c = a + (b - a) / 3
        d = a + (b - a) / 3 * 2
        if func(vk + c * (xk - vk)) >= func(vk + d * (xk - vk)):
            a = c
        else:
            b = d
    return a


def AGMsDR(x0, iters, func, grad):
    xk = x0.copy()
    vk = x0.copy()
    Ak = 0
    res = [func(xk)]
    for i in range(iters):
        betk = ternary_search_bet(0, 1, xk, vk, func)
        yk = vk + betk * (xk - vk)
        grad_yk = grad(yk)
        hk = ternary_search_h(0, 1e5, func, yk, grad_yk)
        xk = yk - hk * grad_yk

        func_yk = func(yk)
        func_xk = func(xk)
        tmp = (func_yk - func_xk) ** 2 + 2 * Ak * np.linalg.norm(grad_yk) ** 2
        ak = (func_yk - func_xk + (tmp) ** 0.5) / np.linalg.norm(grad_yk) ** 2
        Ak += ak
        vk -= ak * grad_yk
        res.append(func(xk))
    return xk, res
