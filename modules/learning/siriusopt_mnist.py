import sys

sys.path.insert(0, '../input_data_read')

import numpy as np
from siriusopt.descent import descent, ternary_search, Grad_sp, RMS, bfgs
from siriusopt.fast import fast_descent, grad_fast_step, Fast_grad_Nest
from siriusopt.fast import Adam, Adagrad, triangles_1_5, triangles_2_0, Heavy_ball
from siriusopt.stochastic import sgd, sag_yana, sag, magic_sag2, mig
from siriusopt.show import show
import input_data_read as mnist
from time import time
from xres import xk


storage = mnist.loadData()

storage.images = np.asarray(storage.images)
storage.labels = np.asarray(storage.labels)

print(storage.images.shape)
print(storage.labels.shape)

func = lambda x: 0.5 * np.linalg.norm(storage.images @ x - storage.labels) ** 2
gradi = lambda x, i: storage.images[i] * (storage.images[i].T @ x - storage.labels[i])
grad = lambda x: storage.images.T @ (storage.images @ x - storage.labels) * 2
# L = np.trace(storage.images)
L = 19590072744 * 3 #np.amax(np.linalg.eigh(2 * storage.images.T @ storage.images)[0])

print(L)

ka = 7000

x0 = xk
xres, points = sag(ka, x0, gradi, func, storage.labels.size, L, storage.images.shape[0])
# xres, points = sgd(x0, gradi, 200, func, L)

# xres, points = mig(x0, func, grad, gradi, 3, L, 200)

# xres, points = triangles_2_0(x0, func, grad, ka)
# xres, points = Heavy_ball(x0, func, grad, ka)

with open("xres.txt", "w", encoding="utf8") as file:
    print(xres, file=file)
"""
xres, points = fast_descent(x0, func, grad, L, ka)

triangle15_res = triangles_1_5(x, func, grad, ka, L)
print("triangle15", func(triangle15_res[0]), func_test(triangle15_res[0]))

triangle20_res = triangles_2_0(x, func, grad, ka)
print("triangle20", func(triangle20_res[0]), func_test(triangle20_res[0]))

hb_res = Heavy_ball(x, func, grad, ka)
print("heavy ball", func(hb_res[0]), func_test(hb_res[0]))

# grad_fast2 = grad_fast_step(grad, func, ka, x)
ag = Fast_grad_Nest(x, func, grad, L, ka)
print("fastgr_sasha", func(ag[0]), func_test(ag[0]))
"""
show([np.log(points)],  f"res{time()}", labels=["magicsag"])
