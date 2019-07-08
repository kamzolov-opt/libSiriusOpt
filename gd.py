import numpy as np

def gradient_step(x_k, grad, step):
    return x_k - step * grad(x_k)

def newton_step(x_k, grad, hessian):
    H = np.linalg.inv(hessian)
    return x_k - H @ grad(x_k)

def demph_newton_step(x_k, grad, hessian, step):
    H = np.linalg.inv(hessian)
    return x_k - step * H @ grad(x_k) 