def Adam(func, grad, x0, steps, alf=1e-1, b1=0.9, b2=0.99, eps=1e-8):
    xk = x0.copy()
    m_t = 0
    v_t = 0
    pow_b1 = b1
    pow_b2 = b2
    res = [func(xk)]
    for i in range(1, steps + 1):
        grad_t = grad(xk)
        m_t = b1 * m_t + (1 - b1) * grad_t
        v_t = b2 * v_t + (1 - b2) * grad_t * grad_t
        m_t_tmp = m_t / (1 - pow_b1)
        v_t_tmp = v_t / (1 - pow_b2)
        pow_b1 = pow_b1 * b1
        pow_b2 = pow_b2 * b2
        xk -= alf * m_t_tmp / (v_t_tmp ** 0.5 + eps)
        res.append(func(xk))
    return xk, res
