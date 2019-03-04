import numpy as np
import math


def sig(x):
    return 1./(1 + np.exp(-x))


def a1(x, y):
    return sig(4*x + 5*y - 9)


print(a1(1, 1))
print(a1(2, 4))
print(a1(5, -5))
print(a1(-4, 5))