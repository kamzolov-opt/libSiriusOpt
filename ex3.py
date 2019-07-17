import numpy as np

from siriusopt.descent import descent, ternary_search, Grad_sp, RMS, bfgs
from siriusopt.fast import fast_descent, grad_fast_step, Fast_grad_Nest
from siriusopt.fast import Adam, Adagrad, triangles_1_5, triangles_2_0, Heavy_ball
from siriusopt.stochastic import sgd, sag_yana, sag, magic_sag2, mig
from siriusopt.show import show

dat_file = np.load('data/student.npz')
A_learn = dat_file['A_learn']
b_learn = dat_file['b_learn']
A_test = dat_file['A_test']
b_test = dat_file['b_test']

m = 395  # number of read examples (total:395)
n = 27  # features
m_learn = 300
x = np.zeros(28)
dodes = []
dodes_x = []
test_r = []
i = np.random.randint(28)

# Функции
func = lambda x: 0.5 * np.linalg.norm(A_learn @ x - b_learn) ** 2


# func_test = lambda x: 0.5 * np.linalg.norm(A_test @ x - b_test) ** 2
def func_test(x):
    predict = A_test @ x
    er = np.sum((predict - b_test) ** 2) / len(b_test)
    return er


grad = lambda x: A_learn.T @ (A_learn @ x - b_learn) * 2
func_h = lambda xk, h, func, grad: func(xk - h * grad)
gradi = lambda x, i: 2 * A_learn[i] * (A_learn[i].T @ x - b_learn[i])
L = np.amax(np.linalg.eigh(2 * A_learn.T @ A_learn)[0])

ka = 2000

valid = 373.40401555816754 - 1e-6

print("Число итераций", ka)
print("Метод", "f(x)", "f_test(x)")
z = descent(x, func, grad, func_h, ka)
print("descent line search", func(z[0]), func_test(z[0]))
ts = Grad_sp(x, func, grad, L, t=ka)
print("descent", func(ts[0]), func_test(ts[0]))

adam = Adam(func, grad, x, ka)
print("adam", func(adam[0]), func_test(adam[0]))

rms_res = RMS(x, 0.003, 0.9, 1, ka, grad, func)
print("rms", func(rms_res[0]), func_test(rms_res[0]))

adagrad_res = Adagrad(x, grad, func, ka)
print("adagard", func(adagrad_res[0]), func_test(adagrad_res[0]))

print(adam[1])
adam_res = [adam[1][0] - valid]
for i in range(1, len(adam[1])):
    adam_res.append(min(adam[1][i] - valid, adam_res[i-1]))
print(adam_res)
# bfgs_res = bfgs(x, ka, func, grad)

# show([np.log(fastgr[1])], namefile="img/ex4", title="Быстрый градиентный спуск")
show([np.log(np.asarray(z[1]) - valid),
      np.log(list(np.asarray(ts[1]) - valid)),
      np.log(adam_res),
      np.log(list(np.asarray(adagrad_res[1]) - valid)),
      np.log(list(np.asarray(rms_res[1]) - valid))],
     namefile="img/ex3",
     labels=["Gradient descent with linear search", "Gradient descent", "Adam", "Adagrad", "RMS"])
