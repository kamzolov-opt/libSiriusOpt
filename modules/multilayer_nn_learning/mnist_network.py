import numpy as np
from siriusopt.show import show
import sys

sys.path.insert(0, '../input_data_read')

import numpy as np
from siriusopt.descent import descent, ternary_search, Grad_sp, RMS, bfgs
from siriusopt.fast import fast_descent, grad_fast_step, Fast_grad_Nest
from siriusopt.fast import Adam, Adagrad, triangles_1_5, triangles_2_0, Heavy_ball
from siriusopt.stochastic import sgd, sag_yana, sag, magic_sag2, mig
from siriusopt.show import show
import input_data_read as mnist
from modules.multilayer_nn_learning.actfuncs import ActivationFuncs
from modules.multilayer_nn_learning.siriusoptnetwork import NeuralNetwork, Config
from itertools import cycle
from time import time

storage = mnist.loadData()
from b_train_normal import res_correct
from weights import weights

storage.images = np.asarray(storage.images)
storage.labels = np.asarray(storage.labels)

nn = NeuralNetwork()
nn.add_layer()
nn.add_tensors(storage.images.shape[1])
nn.add_bias()
nn.add_layer()
nn.add_tensors(10)
nn.unpackParameterFromVector(np.asarray(weights))

"""
print(">>> Training process start")
for i in range(1):
    n, m = 0, 100  # next(nc), next(mc)
    print(f"Training epoc {i}")
    nn.train(iter(storage.images[n:m]), iter(res_correct[n:m]), speed=0.005)
print("Training process end \n")
nn.save()
# Testing
print(">>> Testing process start")
res_test = nn.test(storage.images, res_correct, limit=10)

print("> Testing results")
print(f"All: {res_test[0]}")
print(f"Valid: {res_test[1]}")
print(f"Accuracy: {res_test[1] / res_test[0] * 100} %")
"""

Config.a_learn = storage.images[:100]
Config.b_learn = res_correct[:100]
L = 2000
xk, points = sgd(x0=nn.packParameterToVector(), grad=nn.empiricalRiskGradient,
                 steps=5, func=nn.empiricalRisk, L=L)

show([points], "sgd_ml", labels=["sgd"])