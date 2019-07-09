import numpy as np

class Sag:

    def __init__(self, func, func_full, gradient):
        self.func = func
        self.func_full = func_full
        self.gradient = gradient

        self.values = []
        self.gradients = []

    def init_data(self, A_learn, b_learn, A_test, b_test, count_grads=None):
        count_grads = A_learn.shape[0] if count_grads is None else count_grads
        self.A_learn = A_learn
        self.b_learn = b_learn
        self.A_test = A_test
        self.b_test = b_test

        self.xk = np.ones(A_learn.shape[1])
        for j in [np.random.randint(0, count_grads) for _ in range(self.A_learn.shape[0])]:
            self.gradients.append(self.gradient(self.xk, j))

        self.h = 0.125 / np.amax(np.linalg.eigh((self.A_learn.T @ self.A_learn))[0])
        self.s = sum(self.gradients)

    def step(self, repeat=1):
        for k in range(repeat):
            j = np.random.randint(0, self.A_learn.shape[0] - 1)
            self.s -= self.gradients[j]
            del self.gradients[j]
            yj = self.gradient(self.xk, j)
            self.gradients.append(yj)
            self.xk = self.xk - self.h * (self.s / len(self.s))
            self.s += yj
            self.values.append(self.func_full(self.xk))
            print(self.func_full(self.xk))

    def get_data(self):
        return self.values
