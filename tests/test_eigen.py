import context
from matrix import Matrix
from equation.iter_process.eigen_things import EigenThings

import matplotlib.pyplot as plt

A = Matrix(3, elems = [-7, -6,  8,
                       -6,  3, -7,
                        8, -7,  4]
)

error = 2**(-20)
(values, vectors), iterations = EigenThings(A, error).rotation_method()

print('Error:', error)
print()

print("Eigenvalues:")
for v in values:
    print(v)
print()

print("Eigenvectors:")
for v in vectors:
    print(v, '\n')

print("Check:")
for val, vec in zip(values, vectors):
    print("--A * eigenvector:")
    print(A * vec, '\n')
    print("--eigenvalue * eigenvector")
    print(vec * val, '\n')

errors = [2**i for i in range(5, -15, -1)]
x = []
y = []

for error in errors:
    y.append(error)
    x.append(EigenThings(A, error).rotation_method().iterations)

plt.plot(x, y)
plt.xlabel('iterations')
plt.ylabel('error')
plt.title('Error -- Number of iterations\n dependece')

plt.show()
