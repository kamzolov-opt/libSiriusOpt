import numpy as np
from siriusopt.show import show
import sys, os

dirScript = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirScript)
sys.path.append(os.path.join(dirScript, "./../../"))
sys.path.append(os.path.join(dirScript, "./../../siriusopt"))
# sys.path.append(os.path.join(dirScript, "./../../siriusopt/fast"))

sys.path.insert(0, '../input_data_read')

import numpy as np
from siriusopt.descent import descent, ternary_search, Grad_sp, bfgs
from siriusopt.fast import fast_descent, Fast_grad_Nest
from siriusopt.fast import Adam, Adagrad, triangles_1_5, triangles_2_0, Heavy_ball
from siriusopt.stochastic import sgd, sag, magic_sag2, mig, magic_sag
from siriusopt.show import show
import input_data_read as mnist
from modules.multilayer_nn_learning.actfuncs import ActivationFuncs
from modules.multilayer_nn_learning.siriusoptnetwork import NeuralNetwork, Config
from itertools import cycle
from time import time

storage = mnist.loadData()
from b_train_normal import res_correct
from modules.multilayer_nn_learning.weights import weights

storage.images = np.asarray(storage.images)
storage.labels = np.asarray(storage.labels)

nn = NeuralNetwork()
nn.add_layer()
nn.add_tensors(storage.images.shape[1])
nn.add_bias()
nn.add_layer()
nn.add_tensors(10)
nn.unpackParameterFromVector(np.asarray(weights))

nn.run(storage.images[0])

time_train = time()
print(">>> Training process start")
for i in range(1):
    n, m = 0, 5000  # next(nc), next(mc)
    print(f"Training epoc {i}")
    nn.train(iter(storage.images[n:m]), iter(res_correct[n:m]), speed=0.005)
print(f"Training process end {time() - time_train} s\n")
nn.save()
# Testing
print(">>> Testing process start")
res_test = nn.test(storage.images, res_correct, limit=1000)

print("> Testing results")
print(f"All: {res_test[0]}")
print(f"Valid: {res_test[1]}")
print(f"Accuracy: {res_test[1] / res_test[0] * 100} %")

"""
Config.a_learn = storage.images[:100]
Config.b_learn = res_correct[:100]
L = 2000
weights = nn.packParameterToVector().copy()

xk, points_sgd = sgd(x0=weights.copy(), gradi=nn.empiricalRiskGradient,
                     steps=5, func=nn.empiricalRisk, L=L, num_func=Config.a_learn.shape[0])

xk, points_hb = Heavy_ball(x0=weights.copy(), grad=nn.empiricalRiskGradient, steps=5, func=nn.empiricalRisk)

weights = nn.packParameterToVector().copy()

xk, points_adam = Adam(x0=weights,
                       grad=nn.empiricalRiskGradient, steps=5,
                       func=nn.empiricalRisk)

xk, points_magicsag = magic_sag(x0=weights.copy(),
                                steps=5,
                                gradi=nn.empiricalRiskGradient,
                                L=L,
                                A_shape_0=5,
                                func=nn.empiricalRisk,
                                func_counts=5)

xk, points_triangles15 = triangles_1_5(x0=weights.copy(),
                                       steps=5,
                                       grad=nn.empiricalRiskGradient,
                                       L=L,
                                       func=nn.empiricalRisk)
func_h = lambda xk, h, func, grad: func(xk - h * grad)

xk, points_fastgr = descent(x0=weights.copy(),
                                       steps=5,
                                       grad=nn.empiricalRiskGradient,
                                       func_h=func_h,
                                       func=nn.empiricalRisk)

show([points_sgd, points_magicsag, points_hb, points_triangles15, points_fastgr], "sgd_ml",
     labels=["sgd", "magic_sag", "heavy_ball", "triangles 1.5", "gradient + ls"])
"""