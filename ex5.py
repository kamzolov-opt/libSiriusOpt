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
gradi = lambda x, i: A_learn[i] * (A_learn[i].T @ x - b_learn[i])
L = np.amax(np.linalg.eigh(2 * A_learn.T @ A_learn)[0])

print(L)
ka = 1000
valid = 373.40401555816754 - 1e-6

print("Число итераций", ka)
print("Метод", "f(x)", "f_test(x)")
fastgr = fast_descent(x, func, grad, L, ka)
print("fastgr", func(fastgr[0]), func_test(fastgr[0]))
triangle15_res = triangles_1_5(x, func, grad, ka, L)
print("triangle15", func(triangle15_res[0]), func_test(triangle15_res[0]))

triangle20_res = triangles_2_0(x, func, grad, ka)
print("triangle20", func(triangle20_res[0]), func_test(triangle20_res[0]))

hb_res = Heavy_ball(x, func, grad, ka)
print("heavy ball", func(hb_res[0]), func_test(hb_res[0]))

# grad_fast2 = grad_fast_step(grad, func, ka, x)
ag = Fast_grad_Nest(x, func, grad, L, ka)
print("fastgr_sasha", func(ag[0]), func_test(ag[0]))

# show([np.log(fastgr[1])], namefile="img/ex4", title="Быстрый градиентный спуск")
show([np.log(list(np.asarray(fastgr[1]) - valid)),
      np.log(list(np.asarray(triangle15_res[1]) - valid)),
      np.log(list(np.asarray(triangle20_res[1]) - valid)),
      np.log(list(np.asarray(hb_res[1]) - valid)),
      np.log(list(np.asarray(ag[1]) - valid))], namefile="img/ex5",
     labels=["Triangles 1.0", "Triangles 1.5", "Triangles 2.0", "Heavy ball", "Nesterov method"])
