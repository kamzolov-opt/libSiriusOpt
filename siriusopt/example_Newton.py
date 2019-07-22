import numpy as np
from show import show

from Newton_methods.Newton import Newton
from Newton_methods.Quasi_Newton_v1 import Quasi_Newton_v1

dat_file = np.load('data/student.npz')
A_learn = dat_file['A_learn']
b_learn = dat_file['b_learn']
A_test = dat_file['A_test']
b_test = dat_file['b_test']

m = 395  # number of read examples (total:395)
n = 27  # features
m_learn = 300
x = np.zeros(28)
i = np.random.randint(28)

# Функции
func = lambda x: 0.5 * np.linalg.norm(A_learn @ x - b_learn) ** 2
grad = lambda x: A_learn.T @ (A_learn @ x - b_learn)
func_h = lambda xk, h, func, grad: func(xk - h * grad)
gradi = lambda x, i: A_learn[i] * (A_learn[i].T @ x - b_learn[i])
grad2 = lambda x: A_learn.T @ A_learn
grad2i = lambda x, i: A_learn[i].T @ A_learn[i]
L = np.amax(np.linalg.eigh(2 * A_learn.T @ A_learn)[0])


def func_test(x):
    predict = A_test @ x
    er = np.sum((predict - b_test) ** 2) / len(b_test)
    return er


ka = 3500
valid = 373.4040155581674 - 1e-6

print("Число итераций", ka)
print("Метод", "f(x)", "f_test(x)")

Newton = Newton(x, func, grad, grad2, ka, L)
print("Newton", func(Newton[0]), func_test(Newton[0]))

Quasi_Newton = Quasi_Newton_v1(x, func, grad, grad2, ka, L)
print("Quasi Newton", func(Quasi_Newton[0]), func_test(Quasi_Newton[0]))

show([np.log(list(np.asarray(Newton[1]) - valid)),
      np.log(list(np.asarray(Quasi_Newton[1]) - valid))], namefile="img/Newton",
     labels=["Newton", "Quasi Newton"])
