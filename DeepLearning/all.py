import numpy as np
import math


def sig(x):
    return 1./(1 + np.exp(-x))


def a1(x, y, z):
    return sig(0.4*x + 0.6*y + z)


print(a1(2, 6, -2))
print(a1(3, 5, -2.2))
print(a1(5, 4, -3))