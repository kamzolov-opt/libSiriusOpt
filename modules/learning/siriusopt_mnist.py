import sys

sys.path.insert(0, '../input_data_read')

import numpy as np
from siriusopt.stochastic import sgd, mig
from siriusopt.show import show
import input_data_read as mnist
from time import time


storage = mnist.loadData()

storage.images = np.asarray(storage.images)
storage.labels = np.asarray(storage.labels)

print(storage.images.shape)
print(storage.labels.shape)

func = lambda x: 0.5 * np.linalg.norm(storage.images @ x - storage.labels) ** 2
gradi = lambda x, i: storage.images[i] * (storage.images[i].T @ x - storage.labels[i])
grad = lambda x: storage.images.T @ (storage.images @ x - storage.labels) * 2
L = np.trace(storage.images)
print(L)

x0 = np.zeros(storage.images.shape[1])
# xres, points = sag(200, x0, gradi, func, storage.labels.size, L, storage.images.shape[0])
# xres, points = sgd(x0, gradi, 200, func, L)

xres, points = mig(x0, func, grad, gradi, 3, L, 200)

show([np.log(points)],  f"res{time()}", labels=["mig"])
