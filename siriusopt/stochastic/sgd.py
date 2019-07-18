import numpy as np


def sgd(x0, grad, steps, func, L):
    res = [func(x0)]
    x = x0
    for j in range(1, steps):
        i = np.random.randint(300)
        x -= x - grad(x, i) / L
        res.append(func(x))
    return x, res
