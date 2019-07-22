import numpy as np


def L_FBGS(x0, func, grad, grad_i, grad_2_i, steps, num_func, b=50, bh=600, l=20, nu=1e-5):
    xk = x0.copy()
    res = [func(xk)]
    r = 0
    hr = np.eye(len(xk))
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
