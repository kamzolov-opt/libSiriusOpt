import numpy as np
from show import show
from sag import Sag


func = lambda x, i: 0.5 * (A_learn[i].T @ x - b_learn[i]) ** 2
func_full = lambda x: 0.5 * (A_learn @ x - b_learn).T @ (A_learn @ x - b_learn)
gradfunc = lambda x, j: (A_learn[j] @ x - b_learn[j]) * A_learn[j]

dat_file = np.load('data/student.npz')
#A_learn = dat_file['A_learn']
#b_learn = dat_file['b_learn']
A_learn = np.array([[n+m+1 for n in range(4)] for m in range(5)])
b_learn = np.array([n+1 for n in range(5)])
print(A_learn.shape)
print(b_learn.shape)

A_test = dat_file['A_test']
b_test = dat_file['b_test']


sag = Sag(func, func_full, gradfunc)
sag.init_data(A_learn, b_learn, A_test, b_test)
sag.step(1000)
show([sag.get_data()], namefile="sag")
