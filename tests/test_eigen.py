import sys
sys.path.append('../matrix')
sys.path.append('../equation_tools')

from matrix import Matrix
from equation_tools import EigenThings

import matplotlib.pyplot as plt

A = Matrix(3, elems = [-7, -6,  8,
                       -6,  3, -7,
                        8, -7,  4]
)

error = 0.000001
values, vectors = EigenThings(A, error).get_things()

print('Error:', error)
print()

print("Eigen values:")
for v in values:
    print(v)
print()

print("Eigen vectors:")
for v in vectors:
    print(v, '\n')

print("Check:")
for val, vec in zip(values, vectors):
    print("--Multiplication:")
    print(A * vec, '\n')
    print("--Value")
    print(val, '\n')

errors = [2**i for i in range(5, -15, -1)]
x = []
y = []

for error in errors:
    y.append(error)
    x.append(EigenThings(A, error).get_iterations())

plt.plot(x, y)
plt.xlabel('iterations')
plt.ylabel('error')
plt.title('Error -- Number of iterations\n dependece')

plt.show()
