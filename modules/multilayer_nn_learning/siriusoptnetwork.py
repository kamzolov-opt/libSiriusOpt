import sys

sys.path.insert(0, '../input_data_read')

import numpy as np
from modules.multilayer_nn_learning.actfuncs import ActivationFuncs


class Config:
    a_learn = None
    b_learn = None


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
        return [float(element.value) for element in self.layers[-1].tensors]

    def train(self, a_train, b_train, speed=0.2, correct_params=True):
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
                if correct_params:
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
            run_results = self.run(a_test[i])
            res = list(map(lambda el: 1 if el >= 0.8 else 0, run_results))
            real = b_test[i]
            if res == real:
                correct_experements += 1

            all_experements += 1

        return all_experements, correct_experements, correct_experements / all_experements * 100

    def empiricalRisk(self, wk):
        self.unpackParameterFromVector(wk)
        a_train = Config.a_learn
        b_train = Config.b_learn
        error = 0
        for ex in range(len(a_train)):
            output = self.run(a_train[ex])

            error += sum([(output[i] - b_train[ex][i]) ** 2 / 2 for i in range(len(output))])
        return error

    def empiricalRiskGradient(self, wk, i=0):
        self.unpackParameterFromVector(wk)
        a_train = Config.a_learn
        b_train = Config.b_learn
        self.train(iter(a_train), iter(b_train), correct_params=False)
        dweights = self.packDparameterToVector(len(a_train))
        return dweights / a_train.shape[0]

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
        dweights = np.array([])
        for layer in self.layers:
            for tensor in layer.tensors:
                if not tensor.dweights.size:
                    continue
                dweights = np.concatenate((dweights, tensor.dweights))
        return np.array(dweights) / N


    def packParameterToVector(self):
        weights = []
        for layer in self.layers:
            for tensor in layer.tensors:
                weights += list(tensor.weights)
        return np.asarray(weights)

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
