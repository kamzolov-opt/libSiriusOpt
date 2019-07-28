import numpy as np
from modules.multilayer_nn_learning.actfuncs import ActivationFuncs


class NN:

    def __init__(self):
        self.layers = []

    def forward(self, values):
        self.layers[0].set_tensors(values)
        self.layers[0].activate_first()  # Activation first layer
        for l in range(1, len(self.layers)):
            self.layers[l].activate(self.layers[l - 1].tensors)
        return self.layers[-1].tensors

    def back(self, a_train, b_train, speed=0.2, correct_params=True):
        iteration = 0
        try:
            while True:
                iteration += 1
                # print(f">> training {iteration}")
                # for l in range(1, len(self.layers)):
                #    self.layers[l].dweights = np.zeros(self.layers[l].weights.shape)

                res = self.forward(next(a_train))  # forward propogation
                res = res.reshape(1, res.size)
                b = next(b_train)
                delthas = (res - b) * self.layers[-1].dfunc(res)

                # dw_change = self.layers[-1].tensors.reshape(1, self.layers[-1].tensors.size) * delthas

                # self.layers[-1].dweights += dw_change.T
                self.layers[-1].delthas = delthas

                for l in range(len(self.layers) - 2, 0, -1):
                    deltha_1 = self.layers[l + 1].delthas @ self.layers[l + 1].weights.T
                    deltha_2 = self.layers[l].dfunc(self.layers[l].tensors)
                    delthas = deltha_1.reshape(1, deltha_1.size) * deltha_2.reshape(1, deltha_2.size)
                    self.layers[l].delthas = delthas
                    # dw_change = self.layers[l - 1].tensors.reshape(self.layers[l - 1].tensors.size, 1) * delthas
                    # self.layers[l].dweights += dw_change.T

                if correct_params:
                    for l in range(len(self.layers) - 1, 0, -1):
                        dw = self.layers[l - 1].tensors @ self.layers[l].delthas

                        self.layers[l].weights += -speed * dw

        except StopIteration:
            return

    def add_layer(self, count, bias=False):
        if self.layers:
            layer = Layer(size=count, past_size=self.layers[-1].tensors.size, bias=bias)
        else:
            layer = Layer(size=count + int(bias), past_size=0, bias=bias)

        self.layers.append(layer)

    def ExportParamsToVector(self):
        weights = np.array([])
        for l in range(1, len(self.layers)):
            a = self.layers[l].weights.reshape(1, self.layers[l].weights.size)[0]
            weights = np.concatenate((weights, a))
        return weights

    def ImportParamsFromVector(self, vector):
        for l in range(1, len(self.layers)):
            self.layers[l].weights = vector[:self.layers[l].weights.size].reshape(self.layers[l].weights.shape)
            vector = vector[self.layers[l].weights.size:]
        return True

    def test(self, a_test, b_test, limit=None):

        if limit is None:
            limit = len(a_test)
        all_experements, correct_experements = 0, 0
        for i in range(limit):
            run_results = self.forward(a_test[i])
            res = list(map(lambda el: 1 if el >= 0.8 else 0, run_results))
            real = b_test[i]
            if res == real:
                correct_experements += 1

            all_experements += 1

        return all_experements, correct_experements, correct_experements / all_experements * 100

    def save(self, filename):
        # Save weights
        with open(f"{filename}.py", "w", encoding="utf8") as file:
            file.write("weights = " + str(list(map(lambda x: x, self.ExportParamsToVector()))))


class Layer:

    def __init__(self, size, past_size, bias):

        self.size = (size, past_size)
        self.tensors = np.ones(size).reshape(size, 1)
        if past_size:
            self.weights = np.random.sample(past_size * size).reshape(past_size, size)
            self.dweights = np.zeros(past_size * size).reshape(self.weights.shape)
        self.func = ActivationFuncs.SIGMOID
        self.dfunc = ActivationFuncs.get_derivative(self.func)
        self.bias = bias
        if bias:
            self.tensors[-1] = 1

    def set_tensors(self, values):
        values = np.asarray(values)
        if self.bias:
            values = np.concatenate((values, np.array([1])))
        values = values.reshape(self.tensors.shape)
        self.tensors = values

    def activate(self, parents_tensors):
        self.tensors = self.func(self.weights.T @ parents_tensors)
        if self.bias:
            self.tensors[-1] = 1

    def activate_first(self):
        self.tensors = self.func(self.tensors)
        if self.bias:
            self.tensors[-1] = 1


import modules.input_data_read.input_data_read as mnist

storage = mnist.loadData()
from modules.multilayer_nn_learning.b_train_normal import res_correct
from modules.multilayer_nn_learning.weights_fast import weights
from time import time

data_time = time()
storage.images = np.asarray(storage.images)
storage.labels = np.asarray(storage.labels)

nn = NN()
nn.add_layer(count=storage.images.shape[1], bias=True)
nn.add_layer(count=10)
nn.ImportParamsFromVector(np.asarray(weights))


print(f"Данные загружены {time() - data_time} s")
print(">>> Training process start")
data_time = time()
for i in range(1):
    n, m = 0, 5000  # next(nc), next(mc)
    print(f"Training epoc {i}")
    nn.back(iter(storage.images[n:m]), iter(res_correct[n:m]), speed=0.005, correct_params=True)
print(f"Training process end {time() - data_time} s \n")

res_test = nn.test(storage.images, res_correct, limit=1000)

print("> Testing results")
print(f"All: {res_test[0]}")
print(f"Valid: {res_test[1]}")
print(f"Accuracy: {res_test[1] / res_test[0] * 100} %")

nn.save("weights_fast")