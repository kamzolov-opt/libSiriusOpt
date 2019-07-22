import numpy as np
from show import show

from stochastic.mig import mig
from stochastic.sag import sag, magic_sag, magic_sag2
from stochastic.saga import saga, SAGA_RMS
from stochastic.sgd import sgd

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


ka = 10000
valid = 373.4040155581674 - 1e-6

print("Число итераций", ka)
print("Метод", "f(x)", "f_test(x)")

sag = sag(m_learn, x, gradi, func, L, ka)
print("SAG", func(sag[0]), func_test(sag[0]))

magic_sag = magic_sag(ka, x, gradi, func, n, L, m_learn)
print("Magic SAG", func(magic_sag[0]), func_test(magic_sag[0]))

magic_sag_2 = magic_sag2(ka, x, grad, gradi, func, n, L, m_learn, m_learn)
print("Magic SAG + MiG", func(magic_sag_2[0]), func_test(magic_sag_2[0]))

saga = saga(x, gradi, func, L, ka)
print("SAGA", func(saga[0]), func_test(saga[0]))

saga_rms = SAGA_RMS(x, gradi, func, ka, m_learn, n + 1)
print("SAGA_RMS", func(saga_rms[0]), func_test(saga_rms[0]))

sgd = sgd(x, gradi, ka, func, L, m_learn)
print("SGD", func(sgd[0]), func_test(sgd[0]))

mig = mig(x, func, grad, gradi, ka, L, m_learn)
print("MiG", func(mig[0]), func_test(mig[0]))

show([np.log(list(np.asarray(sag[1]) - valid)),
      np.log(list(np.asarray(magic_sag[1]) - valid)),
      np.log(list(np.asarray(magic_sag_2[1]) - valid)),
      np.log(list(np.asarray(saga[1]) - valid)),
      np.log(list(np.asarray(saga_rms[1]) - valid)),
      np.log(list(np.asarray(sgd[1]) - valid)),
      np.log(list(np.asarray(mig[1]) - valid))], namefile="img/stoch",
     labels=["SAG", "Magic SAG", "Magic SAG + MiG", "SAGA", "SAGA_RMS",
             "SGD", "MiG"])
