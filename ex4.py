import numpy as np

from siriusopt.descent import descent, ternary_search, Grad_sp, RMS, bfgs, SAGAnew
from siriusopt.fast import fast_descent, grad_fast_step, Fast_grad_Nest
from siriusopt.fast import Adam, Adagrad, triangles_1_5, triangles_2_0, Heavy_ball
from siriusopt.stochastic import sgd, sag_yana, sag, magic_sag2, mig, saga
from siriusopt.show import show

dat_file = np.load('data/student.npz')
A_learn = dat_file['A_learn']
b_learn = dat_file['b_learn']
A_test = dat_file['A_test']
b_test = dat_file['b_test']

m = 395  # number of read examples (total:395)
n = 28  # features
m_learn = 300
x = np.zeros(28)
dodes = []
dodes_x = []
test_r = []
i = np.random.randint(28)

# Функции
func = lambda x: 0.5 * np.linalg.norm(A_learn @ x - b_learn) ** 2
#func_test = lambda x: 0.5 * np.linalg.norm(A_test @ x - b_test) ** 2
def func_test(x):
    predict = A_test @ x
    er = np.sum((predict - b_test) ** 2) / len(b_test)
    return er
grad = lambda x: A_learn.T @ (A_learn @ x - b_learn) * 2
func_h = lambda xk, h, func, grad: func(xk - h * grad)
gradi = lambda x, i: 2 * A_learn[i] * (A_learn[i].T @ x - b_learn[i])
L = np.amax(np.linalg.eigh(2 * A_learn.T @ A_learn)[0])

print(L)
ka = 3000
print("Число итераций", ka)
print("Метод", "f(x)", "f_test(x)")
sum_x = sum([np.linalg.norm(A_learn[i] - sum(A_learn[i]) / n) ** 2 for i in range(n)])

sgd_res = sgd(x, gradi, ka, func, L)
print("sgd", func(sgd_res[0]), func_test(sgd_res[0]))

sag_res = sag_yana(m_learn, x, gradi, func, L, ka)
print("sag", func(sag_res[0]), func_test(sag_res[0]))

saga_res = saga(x, gradi, func, L, ka)
print("saga", func(saga_res[0]), func_test(saga_res[0]))

magic_sag = sag(ka, x, gradi, func, 300, L, A_learn.shape[0])
print("magic_sag", func(magic_sag[0]), func_test(magic_sag[0]))

msag2 = magic_sag2(ka, x, grad, gradi, func, 300, L, m_learn, A_learn)
print("magic_sag2", func(msag2[0]), func_test(msag2[0]))


mig_res = mig(x, func, grad, gradi, ka // 300, L, m_learn)
print("mig", func(mig_res[0]), func_test(mig_res[0]))

sagarms_res = SAGAnew(x, gradi, func, ka, m_learn, n)
print("saga_rms", func(sagarms_res[0]), func_test(sagarms_res[0]))

print(mig_res[1])
sol=373.5966856236808
show([np.log(list(np.asarray(sgd_res[1]) - sol)),
      np.log(list(np.asarray(sag_res[1]) - sol)),
      np.log(list(np.asarray(magic_sag[1]) - sol)),
      np.log(list(np.asarray(saga_res[1]) - sol)),
      np.log(list(np.asarray(mig_res[1]) - sol)),
      np.log(list(np.asarray(sagarms_res[1]) - sol)),
      np.log(list(np.asarray(msag2[1]) - sol))], namefile="img/ex4",
     labels=["SGD", "SAG", "Magic_SAG", "SAGA", "MIG", "SAGA+RMS", "magic_sag + mig"])
print(magic_sag[1][-1])
print(mig_res[1][-1])


