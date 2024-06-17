import numpy as np
from load_iris import load_iris

x, y, x_std = load_iris.define_xy()

rgen = np.random.RandomState(None)
r = rgen.permutation(len(y))

sequence = []

for i in range(100):
    sequence.append(i)

print(len(y))
print(" ")
print(type(r))
print(type(x))
print(y[sequence])

a = np.array([[1], [2], [3]])
r = np.array([2, 1, 0])
print(a[r])