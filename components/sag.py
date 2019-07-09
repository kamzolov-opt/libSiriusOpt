import numpy as np
from show import show

class Sag:

    def __init__(self, func, func_full, gradient):
        self.func = func
        self.func_full = func_full
        self.gradient = gradient

        self.xk = np.zeros(28)

        self.values = []
        self.gradients = []

    def init_data(self, A_learn, b_learn, A_test, b_test):
        self.A_learn = A_learn
        self.b_learn = b_learn
        self.A_test = A_test
        self.b_test = b_test

        for j in [np.random.randint(0, 100) for _ in range(301)]:
            self.gradients.append(gradfunc(self.xk, j))

        self.h = 1 / (16 * np.trace(self.A_learn.T @ self.A_learn))
        self.s = sum(self.gradients)

    def step(self, repeat=1):
        for k in range(repeat):
            j = np.random.randint(0, 300)
            self.s -= self.gradients[j]
            del self.gradients[j]
            yj = gradfunc(self.xk, j)
            self.gradients.append(yj)
            self.xk = self.xk - self.h * (self.s / len(self.s))
            self.s += yj
            self.values.append(func_full(self.xk))
            print(func_full(self.xk))

    def get_data(self):
        return self.values


func = lambda x, i: 0.5 * (A_learn[i].T @ x - b_learn[i]) ** 2
func_full = lambda x: 0.5 * (A_learn @ x - b_learn).T @ (A_learn @ x - b_learn)
gradfunc = lambda x, j: (A_learn[j] @ x - b_learn[j]) * A_learn[j]

dat_file = np.load('data/student.npz')
A_learn = dat_file['A_learn']
b_learn = dat_file['b_learn']
A_test = dat_file['A_test']
b_test = dat_file['b_test']

sag = Sag(func, func_full, gradfunc)
sag.init_data(A_learn, b_learn, A_test, b_test)
sag.step(100000)
show([sag.get_data()], namefile="sag")