import numpy as np

def bin_pow_ray(a, b, xk, func, grad_xk):
    if _func_h(xk, b, func, grad_xk) <= _func_h(xk, 2*b, func, grad_xk):
        return ternary_search(a, b, xk, func, grad_xk)
    else:
        return bin_pow_ray(b, 2 * b, xk, func, grad_xk) 
    
def _func_h(xk, h, func, grad_xk):
    return func(xk - h * grad_xk) 

def ternary_search(a, b, xk, func, grad_xk):
    if abs(a - b) < 1e-6:
        return a
    c = a + (b - a) / 3
    d = a + (b - a) / 3 * 2
    if _func_h(xk, c, func, grad_xk) >= _func_h(xk, d, func, grad_xk):
        return ternary_search(c, b, xk, func, grad_xk)
    return ternary_search(a, d, xk, func, grad_xk)