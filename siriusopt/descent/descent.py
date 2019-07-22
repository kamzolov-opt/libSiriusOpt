from siriusopt.gd import gradient_step


def bin_pow_ray(a, b, xk, grad, func_h, func):
    c = (a + b) / 2
    if func_h(xk, c, func, grad) <= func_h(xk, b, func, grad):
        return ternary_search(a, b, xk, grad, func, func_h)
    else:
        return bin_pow_ray(b, 2 * b, xk, grad, func_h, func)


def ternary_search(a, b, xk, grad, func, func_h):
    if abs(a - b) < 1e-6:
        return a
    c = a + (b - a) / 3
    d = a + (b - a) / 3 * 2
    if func_h(xk, c, func, grad) >= func_h(xk, d, func, grad):
        return ternary_search(c, b, xk, grad, func, func_h)
    return ternary_search(a, d, xk, grad, func, func_h)


def descent(x0, func, grad, func_h, steps):
    xk = x0.copy()
    res = [func(xk)]
    for _ in range(steps):
        hk = bin_pow_ray(0, 1, xk, grad(xk), func_h, func)
        xk = gradient_step(xk, grad, hk)
        res.append(func(xk))
    return xk, res


def Grad_sp(x0, func, grad, L, t):
    xk = x0.copy()
    res = [func(xk)]
    for i in range(t):
        xk = gradient_step(xk, grad, 1/L)
        res.append(func(xk))
    return xk, res
