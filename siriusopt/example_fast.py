import numpy as np
from show import show

from fast.adagrad import Adagrad
from fast.AdaGradNorm import Adagrad_norm
from fast.adam import Adam
from fast.fastdescent import fast_descent, Fast_grad_Nest
from fast.heavy_ball import Heavy_ball
from fast.triangle import triangles, triangles_1_5, triangles_2_0

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
grad2i = lambda x, i: A_learn[i].T @ A_learn[i]
L = np.amax(np.linalg.eigh(2 * A_learn.T @ A_learn)[0])


def func_test(x):
    predict = A_test @ x
    er = np.sum((predict - b_test) ** 2) / len(b_test)
    return er


ka = 2000
valid = 373.4040155581674 - 1e-6

print("Число итераций", ka)
print("Метод", "f(x)", "f_test(x)")

#fast_des = fast_descent(grad, func, ka, x, 2 * L)
#print("Fast Descent", func(fast_des[0]), func_test(fast_des[0]))

Nesterov = Fast_grad_Nest(x, func, grad, L, ka)
print("Nesterov", func(Nesterov[0]), func_test(Nesterov[0]))

heavy_ball = Heavy_ball(x, func, grad, ka)
print("Heavy ball", func(heavy_ball[0]), func_test(heavy_ball[0]))

triangle = triangles(x, func, grad, L, ka)
print("Triangles", func(triangle[0]), func_test(triangle[0]))

triangle1 = triangles_1_5(x, func, grad, ka, L)
print("Triangles1.5", func(triangle1[0]), func_test(triangle1[0]))

triangle2 = triangles_2_0(x, func, grad, ka)
print("Triangles2.0", func(triangle2[0]), func_test(triangle2[0]))

adam = Adam(func, grad, x, ka)
print("Adam", func(adam[0]), func_test(adam[0]))

adagrad = Adagrad(x, grad, func, ka)
print("AdaGrad", func(adagrad[0]), func_test(adagrad[0]))

adagrad_norm = Adagrad_norm(x, func, gradi, ka)
print("AdaGradNorm", func(adagrad_norm[0]), func_test(adagrad_norm[0]))

show([#np.log(list(np.asarray(fast_des[1]) - valid)),
      np.log(list(np.asarray(Nesterov[1]) - valid)),
      np.log(list(np.asarray(heavy_ball[1]) - valid)),
      np.log(list(np.asarray(triangle[1]) - valid)),
      np.log(list(np.asarray(triangle1[1]) - valid)),
      np.log(list(np.asarray(triangle2[1]) - valid)),
      np.log(list(np.asarray(adam[1]) - valid)),
      np.log(list(np.asarray(adagrad[1]) - valid)),
      np.log(list(np.asarray(adagrad_norm[1]) - valid))], namefile="img/fast",
     labels=["Nesterov", "Heavy ball", "Triangles", "Triangles1.5", "Triangles2.0",
             "Adam", "AdaGrad", "AdaGradNorm"])
