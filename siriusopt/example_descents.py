import numpy as np
from show import show

from descent.descent import descent, Grad_sp
from descent.bfgs import bfgs
from descent.L_BFGS import L_FBGS
from descent.rms import RMS
from descent.Sub_GRad import Sub_Grad

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


# func_test = lambda x: 0.5 * np.linalg.norm(A_test @ x - b_test) ** 2
def func_test(x):
    predict = A_test @ x
    er = np.sum((predict - b_test) ** 2) / len(b_test)
    return er


grad = lambda x: A_learn.T @ (A_learn @ x - b_learn)
func_h = lambda xk, h, func, grad: func(xk - h * grad)
gradi = lambda x, i: A_learn[i] * (A_learn[i].T @ x - b_learn[i])
grad2i = lambda x, i: A_learn[i].T @ A_learn[i]
L = np.amax(np.linalg.eigh(2 * A_learn.T @ A_learn)[0])

print(L)
ka = 1000
valid = 373.4040155581674 - 1e-6

print("Число итераций", ka)
print("Метод", "f(x)", "f_test(x)")

gd = Grad_sp(x, func, grad, L, ka)
print("Gradient Descent", func(gd[0]), func_test(gd[0]))

gr_ds_with_lnserch = descent(x, func, grad, func_h, ka)
print("Gradient Descent + ln search", func(gr_ds_with_lnserch[0]), func_test(gr_ds_with_lnserch[0]))

bfgs = bfgs(x, func, grad, ka)
print("BFGS", func(bfgs[0]), func_test(bfgs[0]))

l_bfgs = L_FBGS(x, func, grad, gradi, grad2i, 1000 // len(x), m_learn, 50, 600, 20, 0.00001)
print("L_BFGS", func(l_bfgs[0]), func_test(l_bfgs[0]))

rms = RMS(x, grad, func, ka)
print("RMS", func(rms[0]), func_test(rms[0]))

sub_gr = Sub_Grad(x, ka, func, grad)
print("Sub_gr", func(sub_gr[0]), func_test(sub_gr[0]))

show([np.log(list(np.asarray(gd[1]) - valid)),
      np.log(list(np.asarray(gr_ds_with_lnserch[1]) - valid)),
      np.log(list(np.asarray(bfgs[1]) - valid)),
      np.log(list(np.asarray(l_bfgs[1]) - valid)),
      np.log(list(np.asarray(sub_gr[1]) - valid)),
      np.log(list(np.asarray(rms[1]) - valid))], namefile="img/descents",
     labels=["Gradient Descent", "GDLS", "BFGS", "L-BFGS", "Subgr", "RMS"])
