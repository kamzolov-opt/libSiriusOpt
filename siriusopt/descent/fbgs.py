import numpy as np


def bin_search(a, b, pk, xk, grad, func, c1, c2):
    while b - a > 1e-6:
        c = (a + b) / 2
        t = func(xk + c * pk)
        if func(xk + c * pk) > func(xk) + c1 * b * grad(xk).T @ pk:
            b = c
        else:
            a = c
    return b


def bin_search_ray_max(a, b, pk, xk, grad, func, c1, c2):
    c = (a + b) / 2
    if abs(grad(xk + c * pk).T @ pk) > c2 * abs(grad(xk).T @ pk):
        return bin_search(a, c, pk, xk, grad, func, c1, c2)
    else:
        return bin_search_ray_max(c, 2 * b, pk, xk, grad, func, c1, c2)


def bfgs(x0, ka, func, grad, alf=0.1, c1=1e-3, c2=0.999):
    xk = x0.copy()
    res = [func(xk)]
    hk = np.ones((len(xk), len(xk))) * 0.1
    for i in range(ka):
        xk_prev = xk.copy()
        pk = -hk @ grad(xk)
        alf = bin_search_ray_max(0, 1, pk, xk, grad, func, c1, c2)
        xk += alf * pk

        delt_grad = grad(xk) - grad(xk_prev)
        delt_x = xk - xk_prev

        p_not_pk = 1 / (delt_grad.T @ delt_x)
        t = np.outer(delt_grad, delt_x)
        tm = np.eye(len(xk)) - p_not_pk * t
        tmp = np.eye(len(xk)) - p_not_pk * t.T
        hk = tmp @ hk @ tm + p_not_pk * np.outer(delt_x, delt_x)
        res.append(func(xk))

    return ka, res


def L_FBGS(x0, func, grad, grad_i, grad_2_i, steps, num_func, b, bh, l, nu):
    xk = x0.copy()
    res = [func(xk)]
    r = 0
    n = num_func + 1
    hr = np.eye(len(xk))
    grad_first = grad(x0)
    sum_xk = xk
    prev_ur = np.zeros(len(xk))
    omeg = x0.copy()
    for k in range(len(xk)):
        gr = grad(xk)
        xk = omeg
        for t in range(steps):
            s_b = np.random.randint(0, num_func, b)
            grads = [grad_i(xk, i) - grad_i(omeg, i) + gr for i in s_b]
            vt = sum(grads) / b
            xk -= nu * hr @ vt

            if k % l == 0 & k > 0:
                r += 1
                ur = sum_xk / l
                tet_r = np.random.randint(0, num_func, bh)
                grad_2tmp = [grad_2_i(xk, i) for i in tet_r]
                grad_2t = sum(grad_2tmp) / bh
                sr = ur - prev_ur
                yr = grad_2t * sr

                poj = 1 / (sr.T @ yr)
                I = np.eye(len(xk))
                tmp_matr = (I - poj * np.outer(sr, yr))
                hr = tmp_matr.T @ hr @ tmp_matr + poj * np.outer(sr, sr)

                prev_ur = ur
                sum_xk = 0
            sum_xk += xk
            res.append(func(xk))
        omeg[k] = xk[np.random.randint(0, len(xk), 1)]
    return xk, res
