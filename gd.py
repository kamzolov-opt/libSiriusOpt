import numpy as np

def gradient_step(x_k, grad, step):
    return x_k - step * grad(x_k)

def newton_step(x_k, grad, hessian):
    H = np.linalg.inv(hessian)
    return x_k - H @ grad(x_k)

def demph_newton_step(x_k, grad, hessian, step):
    H = np.linalg.inv(hessian)
    return x_k - step * H @ grad(x_k)

def fast_grad_des_v1(xk, x_prev, grad, h, g = 0.8):
    yk = xk - h * grad(xk)
    x_next = yk + g * (xk - x_prev)
    return x_next