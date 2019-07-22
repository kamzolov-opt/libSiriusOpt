import numpy as np


def sgd(x0, gradi, steps, func, L, num_func):
    x = x0.copy()
    res = [func(x)]
    for j in range(1, steps + 1):
        i = np.random.randint(num_func)
        x -= gradi(x, i) / L
        res.append(func(x))
    return x, res
