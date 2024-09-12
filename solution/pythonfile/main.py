import numpy as np
from numpy import linalg as la

A = np.array([[1, 2, 3], [3, 2, 1], [1, 0, -1]])
e, v = la.eig(A)
print(e)
