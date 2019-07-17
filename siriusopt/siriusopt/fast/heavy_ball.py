from siriusopt.gd import fast_grad_des_v1


def Heavy_ball(x0, func, grad, steps, h=0.001):
    xk, xk_prev = x0, x0
    res = [func(xk)]
    for i in range(steps):
        xk_t = fast_grad_des_v1(xk, xk_prev, grad, h)
        xk_prev = xk
        xk = xk_t
        res.append(func(xk))
    return xk, res
