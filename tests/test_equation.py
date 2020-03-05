import sys
sys.path.append('../matrix')
sys.path.append('../equation')

from matrix import Matrix
from equation import Equation

A = Matrix(4, elems = [-7,  3, -4,  7,
                        8, -1, -7,  6,
                        9,  9,  3, -6,
                       -7, -9, -8, -5]
          )
B = Matrix(1, 4, [-126, 29, 27, 34])
B = B.transpose()

eq = Equation(A, B)
X = eq.analytic_solution()

print("Answer:")
print(X)
print()

print("Calculated:")
print(A * X)
print()
print("Given:")
print(B)
