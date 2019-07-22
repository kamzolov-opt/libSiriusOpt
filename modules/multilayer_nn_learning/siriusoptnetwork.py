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
from itertools import cycle
from time import time


class NeuralNetwork:
    sfunc = None

    def __init__(self):
        self.layers = []

    def run(self, values):
        for i in range(len(self.layers[0].tensors) - 1):
            self.layers[0].tensors[i].value = values[i]
        for layer in self.layers:
            for tensor in layer.tensors:
                if not tensor.bias:
                    tensor.activate()
        return [element.value for element in self.layers[-1].tensors]

    def train(self, a_train, b_train, speed=0.2):
        # A_train - train set x
        # b_train - train set y component
        # speed - learning rate
        iteration = 0
        try:
            while True:
                iteration += 1
                if iteration % 100 == 0:
                    print("> Iterration training", iteration)
                self.run(next(a_train))  # forward propogation
                b = next(b_train)
                count_output_tensors = len(b)
                for ex in range(count_output_tensors):
                    valid = b[ex]
                    tensor = self.layers[-1].tensors[ex]
                    deltha = (tensor.value - valid) * tensor.dfunc(tensor.value)
                    self.layers[-1].tensors[ex].deltha = deltha

                    for i in range(tensor.weights.size):
                        out_i = tensor.parents[i].value
                        tensor.dweights[i] += deltha * out_i

                for l in range(len(self.layers) - 2, 0, -1):
                    for t in range(len(self.layers[l].tensors)):
                        tensor = self.layers[l].tensors[t]
                        # if not tensor.bias:
                        deltha = sum(self.get_deltha_of_layer(l + 1) *
                                     self.select_weights(layer=l + 1, tensor=t)) * tensor.dfunc(tensor)
                        self.layers[l].tensors[t].deltha = deltha

                        for i in range(tensor.weights.size):
                            out_i = tensor.parents[i].value
                            tensor.dweights[i] += deltha * out_i

                for l in range(len(self.layers) - 1, 0, -1):
                    for t in range(len(self.layers[l].tensors)):
                        tensor = self.layers[l].tensors[t]
                        self.change_weights(l - 1, tensor, speed, tensor.deltha)

        except StopIteration:
            return

    def change_weights(self, layer, tensor, speed, deltha):
        weights = tensor.weights
        for i in range(len(weights)):
            xn = self.layers[layer].tensors[i].value
            weights[i] += -speed * deltha * xn
        tensor.set_weights(weights)

    def test(self, a_test, b_test, limit=None):
        if limit is None:
            limit = len(a_test)
        all_experements, correct_experements = 0, 0
        for i in range(limit):
            res = list(map(lambda el: 1 if el >= 0.8 else 0, nn.run(a_test[i])))
            real = b_test[i]
            if res == real:
                correct_experements += 1

            all_experements += 1

        return all_experements, correct_experements, correct_experements / all_experements * 100

    def add_layer(self):
        self.layers.append(Layer())

    def add_tensors(self, count=1, act_func=None, bias=False):
        if act_func is None:
            act_func = ActivationFuncs.SIGMOID
        for i in range(count):
            if len(self.layers) >= 2:
                self.layers[-1].add_tensor(act_func=act_func, layer_past=self.layers[-2], bias=bias)
            elif len(self.layers) == 1:
                self.layers[-1].add_tensor(act_func=act_func, bias=bias)

    def add_bias(self):
        self.add_tensors(count=1, act_func=ActivationFuncs.CONST, bias=True)

    def getZeroParams(self):
        length = 0
        for layer in self.layers:
            for tensor in layer.tensors:
                length += tensor.weights.size
        return np.zeros((length, 1))

    def reset_dweights(self):
        for layer in self.layers:
            for tensor in layer.tensors:
                tensor.dweights = np.zeros(tensor.weights.shape)
        return

    def packDparameterToVector(self, N):
        dweights = None
        for layer in self.layers:
            for tensor in layer.tensors:
                if dweights is None:
                    dweights = tensor.dweights.copy()
                dweights += tensor.dweights
        return np.array(dweights) / N

    def packParameterToVector(self):
        weights = []
        for layer in self.layers:
            for tensor in layer.tensors:
                weights += list(tensor.weights)
        return np.asarray(weights).reshape((len(weights), 1))

    def empiricalRiskGradient(self):
        dweights = []
        for layer in self.layers:
            for tensor in layer.tensors:
                dweights += list(tensor.weights)
        return np.asarray(dweights).reshape((len(dweights), 1))

    def unpackParameterFromVector(self, vector):
        index = 0
        for layer in self.layers:
            for tensor in layer.tensors:
                tensor.weights = vector[index:index + len(tensor.parents)]
                index += len(tensor.parents)
        return True

    def get_deltha_of_layer(self, layer):
        return np.asarray([n.deltha for n in self.layers[layer].tensors])

    def select_weights(self, layer, tensor):
        # return np.asarray([n.get_parents_values()[tensor] for n in self.layers[layer].tensors])
        # return self.layers[layer].tensors[tensor].weights
        return np.asarray([n.weights[tensor] for n in self.layers[layer].tensors])

    def save(self):
        # Save weights
        with open("weights.py", "w", encoding="utf8") as file:
            file.write("weights = " + str(list(map(lambda x: x[0], self.packParameterToVector()))))


class Layer:

    def __init__(self):
        self.tensors = []

    def add_tensor(self, act_func, layer_past=None, bias=False):
        if layer_past:
            parents = layer_past.tensors
            weights = np.random.sample(len(parents))
            tensor = Tensor(parents, weights, act_func, bias)
        else:
            tensor = Tensor([], [], act_func, bias)
        self.tensors.append(tensor)


class Tensor:

    def __init__(self, parents, weights, act_func, bias):
        self.parents = parents
        self.weights = np.asarray(weights)  # Each weight correspond to each parent
        self.value = 1  # Actvation or Output of neuron
        self.act_func = act_func
        self.dfunc = ActivationFuncs.get_derivative(self.act_func)
        self.dweights = np.zeros(self.weights.shape)
        self.bias = bias

    def activate(self):
        if not self.parents:
            self.value = self.act_func(self.value)
            return
        values = self.get_parents_values()

        self.value = self.act_func(sum(values * self.weights))

    def set_weights(self, weights):
        self.weights = weights

    def get_parents_values(self):
        return np.array(list(map(lambda element: element.value, self.parents)))


if __name__ == '__main__':

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

    print(">>> Training process start")
    for i in range(1):
        n, m = 0, 10000  # next(nc), next(mc)
        print(f"Training epoc {i}")
        nn.train(iter(storage.images[n:m]), iter(res_correct[n:m]), speed=0.005)
    print("Training process end \n")
    nn.save()

    # Testing
    print(">>> Testing process start")
    res_test = nn.test(storage.images, res_correct, limit=1000)

    print("> Testing results")
    print(f"All: {res_test[0]}")
    print(f"Valid: {res_test[1]}")
    print(f"Accuracy: {res_test[1] / res_test[0] * 100} %")
