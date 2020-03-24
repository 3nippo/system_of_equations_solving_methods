import sys
sys.path.append('../matrix')
sys.path.append('../equation_tools')

from matrix import Matrix
from equation_tools import EigenThings

A = Matrix(3, elems = [-7, -6,  8,
                       -6,  3, -7,
                        8, -7,  4]
)

error = 0.000001
values, vectors = EigenThings(A, error).get_things()

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
