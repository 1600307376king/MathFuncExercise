from sympy import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative
import random


class TaylorSeriesFunction(object):
    """

    """
    def __init__(self, func, series_len, x0):
        if series_len < 1:
            raise AttributeError("series_len can't be less than %s" % series_len)
        self.series_length = series_len
        self.func_obj = func
        self.x0 = x0
        self.derivative_val = 0
        self.taylor_val = None
        self.factorial_val = 1

    def get_factorial(self, num):
        self.factorial_val = 1
        if num > 0:
            for i in range(1, num + 1):
                self.factorial_val *= num
        return self.factorial_val

    def get_derivative_val(self, n):
        self.derivative_val = self.func_obj(self.x0)
        if n > 0:
            for i in range(n - 1):
                derivative(self.func_obj, self.x0, dx=1e-6)
            self.derivative_val = derivative(self.func_obj, self.x0, dx=1e-6)
        return self.derivative_val

    # def lagrangian_remainder(self, x):
    #     self.xi = random.uniform(self.x0, x)
    #     return 1 / (self.get_factorial(self.series_length + 1)) * ((x - self.x0) ** self.series_length + 1) * \
    #         self.get_derivative_val(self.xi, self.series_length + 1)

    def compute_taylor_series_val(self, x):
        self.taylor_val = np.zeros((len(x), ))
        for i in range(self.series_length + 1):
            self.taylor_val += 1 / (self.get_factorial(i)) * ((x - self.x0) ** i) * self.get_derivative_val(i)

        return self.taylor_val


def test_func(x1):
    return x1 ** 2 + x1


ty = TaylorSeriesFunction(test_func, 3, x0=2)
a1 = np.array(range(-10, 10))
b1 = test_func(a1)
b2 = ty.compute_taylor_series_val(a1)

plt.plot(a1, b1, color='b')
plt.plot(a1, b2, color='r')
plt.show()
