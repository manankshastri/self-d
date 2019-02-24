import numpy as np

region = np.array([[1, 2, 3],[1,2, 3], [1, 2, 3]])

sx = np.array([[-1, 0, 1],[-2,0, 2], [-1, 0, -1]])
sy = sx.T
z = region * sx

print(z)
