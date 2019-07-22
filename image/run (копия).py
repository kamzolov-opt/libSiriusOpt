#!/usr/bin/env python3

"""Total variation denoising."""

import argparse
import time

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from scipy.linalg import blas

EPS = np.finfo(np.float32).eps


def dot(x, y):
    """Returns the dot product of two arrays with the same shape."""
    return blas.sdot(x.reshape(-1), y.reshape(-1))

dot(np.array(0), np.array(0))


def axpy(a, x, y):
    """Sets y = a*x + y and returns y."""
    shape = x.shape
    x, y = x.reshape(-1), y.reshape(-1)
    return blas.saxpy(x, y, a=a).reshape(shape)

axpy(1, np.array(0), np.array(0))


def tv_norm(x):
    """Computes the total variation norm and its gradient. From jcjohnson/cnn-vis."""
    x_diff = x - np.roll(x, -1, axis=1)
    y_diff = x - np.roll(x, -1, axis=0)
    grad_norm2 = x_diff**2 + y_diff**2 + EPS
    norm = np.sum(np.sqrt(grad_norm2))
    dgrad_norm = 0.5 / np.sqrt(grad_norm2)
    dx_diff = 2 * x_diff * dgrad_norm
    dy_diff = 2 * y_diff * dgrad_norm
    grad = dx_diff + dy_diff
    grad[:, 1:, :] -= dx_diff[:, :-1, :]
    grad[1:, :, :] -= dy_diff[:-1, :, :]
    return norm, grad


def l2_norm(x):
    """Computes 1/2 the square of the L2-norm and its gradient."""
    return np.sum(x**2) / 2, x


class LBFGSOptimizer:
    """Implements the L-BFGS quasi-Newton optimizer."""
    def __init__(self, params, opfunc, step_size=1, n_corr=10, c1=1e-4, c2=0.9, max_ls_fevals=10):
        """Initializes the optimizer."""
        self.params = params
        self.opfunc = opfunc
        self.step_size = step_size
        self.n_corr = n_corr
        self.c1 = c1
        self.c2 = c2
        self.max_ls_fevals = max_ls_fevals
        self.step = 0
        self.fevals = 0
        self.loss = None
        self.grad = None
        self.sk = []
        self.yk = []

    def update(self):
        """Returns a step's parameter update."""
        self.step += 1

        if self.step == 1:
            self.loss, self.grad = self.opfunc(self.params)
            self.fevals += 1

        # Line search.
        step_size, step_min, step_max = 1, 0, np.inf
        ls_fevals = 0
        while True:
            if ls_fevals == self.max_ls_fevals:
                raise RuntimeError('Gave up on line search')

            # Compute search direction, step, loss, and gradient
            p = -self.inv_hv(self.grad)
            s = step_size * p
            loss, grad = self.opfunc(self.params + s)
            self.fevals += 1
            y = grad - self.grad
            ls_fevals += 1

            # Test that the weak Wolfe curvature condition holds
            if dot(p, grad) < self.c2 * dot(p, self.grad):
                step_min = step_size
            # Test that the Armijo condition holds
            elif loss > self.loss + self.c1 * step_size * dot(p, self.grad):
                step_max = step_size
                self.store_curvature_pair(s, y)
            # Both hold, accept the step
            else:
                break

            # Compute new step size
            if step_max < np.inf:
                step_size = (step_min + step_max) / 2
            else:
                step_size *= 2

        # Update params
        self.params += s

        # Store curvature pair and gradient
        self.store_curvature_pair(s, y)
        self.loss, self.grad = loss, grad
        return loss, self.params

    def store_curvature_pair(self, s, y):
        """Updates the L-BFGS memory with a new curvature pair."""
        self.sk.append(s)
        self.yk.append(y)
        if len(self.sk) > self.n_corr:
            self.sk, self.yk = self.sk[1:], self.yk[1:]

    def inv_hv(self, p):
        """Computes the product of a vector with an approximation of the inverse Hessian."""
        p = p.copy()
        alphas = []
        for s, y in zip(reversed(self.sk), reversed(self.yk)):
            alphas.append(dot(s, p) / (dot(s, y)) + EPS)
            axpy(-alphas[-1], y, p)

        if len(self.sk) > 0:
            s, y = self.sk[-1], self.yk[-1]
            p *= dot(s, y) / (dot(y, y) + EPS)
        else:
            p /= np.sqrt(dot(p, p) / p.size) + EPS

        for s, y, alpha in zip(self.sk, self.yk, reversed(alphas)):
            beta = dot(y, p) / (dot(s, y) + EPS)
            axpy(alpha - beta, s, p)

        return p

def saveimage(namefile, image):
   plt.axis('off')
   plt.imshow(image)
   plt.savefig(namefile)

def corr(namefile, konf):
    #np.random.seed(1)
    u_orig = plt.imread(namefile)
    rows, cols, colors = u_orig.shape
    known = np.zeros((rows, cols, colors))
    for i in range(rows):
        for j in range(cols):
            if np.random.random() > konf:
              for k in range(colors):
                  known[i, j, k] = 1
    u_corr = known * u_orig
    saveimage('new_image', u_corr)
    return 'new_image'

def printf(s, *args, **kwargs):
    print(s.format(*args), **kwargs)


def main():
    """The main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image', help='the input image')
    parser.add_argument('strength', type=float, default=50, nargs='?',
                        help='the denoising strength')
    #parser.add_argument('corr', help = 'the corr image', default = 0.001)
    args = parser.parse_args()

    #namefile = corr(args.input_image, args.corr)
    #args.input_image = namefile

    img = np.float32(Image.open(args.input_image))
    np.random.seed(0)
    img += np.random.normal(scale=30, size=img.shape)
    orig_img = img.copy()

    step_size = 1
    lmbda = args.strength

    def opfunc(img):
        tv_loss, tv_grad = tv_norm(img)
        l2_loss, l2_grad = l2_norm(img - orig_img)
        loss = tv_loss + l2_loss/lmbda
        grad = tv_grad + l2_grad/lmbda
        return loss, grad

    last_loss = np.inf
    steps = 0
    time_start = time.perf_counter()
    print('Optimizing using gradient descent.')
    while True:
        steps += 1
        loss, grad = opfunc(img)
        print('step:', steps, 'loss:', loss)
        if loss > last_loss:
            break
        last_loss = loss
        axpy(-step_size, grad, img)
    time_end = time.perf_counter()

    printf('{} iterations', steps)
    printf('{:g} ms total', 1000 * (time_end - time_start))
    printf('{:g} ms/iteration', 1000 / steps * (time_end - time_start))
    img_gd = img

    img = orig_img.copy()
    opt = LBFGSOptimizer(img, opfunc, n_corr=4)
    last_loss = np.inf
    steps = 0

    time_start = time.perf_counter()
    print('\nOptimizing using L-BFGS.')
    while True:
        steps += 1
        loss, img[:] = opt.update()
        print('step:', steps, 'loss:', loss)
        if loss * 1.01 > last_loss:
            break
        last_loss = loss
    time_end = time.perf_counter()

    printf('{} iterations', steps)
    printf('{} function evaluations', opt.fevals)
    printf('{:g} ms total', 1000 * (time_end - time_start))
    printf('{:g} ms/iteration', 1000 / steps * (time_end - time_start))

    Image.fromarray(np.uint8(np.clip(img_gd, 0, 255))).show()
    Image.fromarray(np.uint8(np.clip(img, 0, 255))).show()

if __name__ == '__main__':
    main()
