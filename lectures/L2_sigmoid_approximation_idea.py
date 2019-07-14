 #!/usr/bin/env python3 
# L2_sigmoid_approximation_idea.py, Konstantin Burlachenko 

import numpy as np
import matplotlib.pyplot as plt
def y_aprox(x):
    if (x < 0):
        compute = 1/(1+(1-x+x*x/2))
        return compute
    else:
        return 1 - y_aprox(-x)
x = np.arange(-20, 20, 0.1)
y = 1/(1+np.exp(-x))
yapr = [y_aprox(xi) for xi in x]
plt.plot(x, y)
plt.plot(x, yapr)
plt.title('Sigmoid and it approximation')
plt.legend(['Sigmoid', 'Sigmoid approximation'])
plt.show()
