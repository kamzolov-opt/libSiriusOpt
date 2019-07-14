#!/usr/bin/env python3
# L2_plot_snippet.py 


import numpy as np
import matplotlib.pyplot as plt
import sys

x = np.arange(0.0, 10.0, 0.1)

#=====================================================
beta = 2.0
y1 = (beta-1.0)*0.5*x*x + (2.0-beta)*np.abs(x)
plt.plot(x, y1, 'g-', label='EN beta=' + str(beta))
#=====================================================

#=====================================================
beta = 1.6
y2 = (beta-1.0)*0.5*x*x + (2.0-beta)*np.abs(x)
plt.plot(x, y2, 'r-', label='EN beta=' + str(beta))
#=====================================================

#=====================================================
beta = 1.1
y3 = (beta-1.0)*0.5*x*x + (2.0-beta)*np.abs(x)
plt.plot(x, y3, 'b-', label='EN beta=' + str(beta))
#=====================================================

yMin = np.min(np.concatenate( (y1, y2, y3) ))
yMax = np.max(np.concatenate( (y1, y2, y3) ))

#=====================================================
plt.xlim(x[0], x[-1])
plt.ylim(x[0], x[-1])
plt.gca().set_aspect('equal', adjustable='box')
#=====================================================

plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend(loc='upper left')
plt.grid(True)
plt.title('Example scalar functions')
plt.show()
