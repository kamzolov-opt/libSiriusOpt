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
from itertools import cycle
from time import time





class NeuralNetwork:
    sfunc = None

    def __init__(self, sfunc, dfunc):
        NeuralNetwork.sfunc = sfunc
        self.layers = []
        self.dfunc = dfunc

    def run(self, values):
        for i, element in enumerate(self.layers[0].tensors):
            element.value = values[i]
        for layer in self.layers:
            for tensor in layer.tensors:
                tensor.activate()
        return [element.value for element in self.layers[-1].tensors]

    def train(self, a_train, b_train, speed=0.2):
        for ex in range(len(a_train)):
            result = self.run(a_train[ex])
            #error = [0.5 * (res - b_train[ex]) ** 2 for res in result]
            #print("training", result, error)
            for valid in b_train[ex]:
                for l, layer in enumerate(self.layers):
                    if l == 0:
                        continue
                    for tensor in layer.tensors:
                        de = (tensor.value - valid) * self.dfunc(tensor.value)
                        weights = tensor.weights
                        for i in range(len(weights)):
                            xn = self.layers[l - 1].tensors[i].value
                            dfunc_dw = de * xn
                            weights[i] += -speed * dfunc_dw
                        tensor.set_weights(weights)

    def test(self, a_test, b_test):
        pass

    def add_layer(self):
        self.layers.append(Layer())

    def add_tensors(self, count=1):
        for i in range(count):
            if len(self.layers) >= 2:
                self.layers[-1].add_tensor(self.layers[-2])
            elif len(self.layers) == 1:
                self.layers[-1].add_tensor()

    def get_all_weights(self):
        weights = []
        for layer in self.layers:
            for tensor in layer.tensors:
                weights += list(tensor.weights)
        return weights


class Layer:

    def __init__(self):
        self.tensors = []

    def add_tensor(self, layer_past=None):
        if layer_past:
            parents = layer_past.tensors
            weights = np.random.sample(len(parents))
            print(len(parents))
            # weights = [1 for _ in range(len(parents))]
            tensor = Tensor(parents, weights)
        else:
            tensor = Tensor([], [])
        self.tensors.append(tensor)


class Tensor:

    def __init__(self, parents, weights, name=None):
        self.parents = parents
        self.weights = weights
        self.value = None
        self.name = name if name else str(self)

    def activate(self):
        if not self.parents:
            self.value = NeuralNetwork.sfunc(self.value)
            return
        values = np.array(list(map(lambda element: element.value, self.parents)))

        self.value = NeuralNetwork.sfunc(sum(values * self.weights))

    def set_weights(self, weights):
        self.weights = weights


if __name__ == '__main__':

    storage = mnist.loadData()

    storage.images = np.asarray(storage.images)
    storage.labels = np.asarray(storage.labels)


    print(storage.images.shape)
    print(storage.labels.shape)

    # func = lambda x: np.log(x + np.sqrt(x ** 2 + 1))
    # dfunc = lambda x: 1 / np.sqrt(x ** 2 + 1)
    func = lambda x: 1.0 / (1.0 + np.exp(-x))
    dfunc = lambda x: func(x) * (1 - func(x))
    func_error = lambda x: 0.5 * np.linalg.norm(storage.images @ x - storage.labels) ** 2



    nn = NeuralNetwork(func, dfunc)
    nn.add_layer()
    nn.add_tensors(storage.images.shape[1])
    nn.add_layer()
    nn.add_tensors(10)

    res = nn.run(storage.images[0])
    print(res)
    res_correct = []
    points = []
    nc, mc = cycle(iter(list(range(0, 1000, 100)))), cycle(iter(list(range(100, 1000, 100))))
    for j in storage.labels[:100]:
        data = [0 if _ != storage.labels[j] else 1 for _ in range(0, 11)]
        res_correct.append(data)

    for i in range(10):
        n, m = 0, 100 # next(nc), next(mc)
        print(i)
        nn.train(storage.images[n:m], res_correct[n:m], speed=0.1)
        # xk = nn.get_all_weights()

        # points.append(func_error(xk))



    show([np.log(points)], "mynn2")
