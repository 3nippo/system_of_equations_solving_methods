import context
from matrix import Matrix
from equation import Equation, LUDecomposition

A = Matrix(4, elems = [-7,  3, -4,  7,
                        8, -1, -7,  6,
                        9,  9,  3, -6,
                       -7, -9, -8, -5]
          )
B = Matrix(1, 4, [-126, 29, 27, 34])
B = B.transpose()

lu = LUDecomposition(A, B)
eq = Equation(A, B, lu)
X = eq.analytic_solution()

print("Answer:")
print(X)
print()

print("Calculated:")
print(A * X)
print()
print("Given:")
print(B)
print()

print("Determinant:")
print(lu.det())
print()

inv = lu.inverse_matrix()
print("Inverse matrix:")
print(inv)
print()

print("Multiplicaion by inverse matrix:")
print(inv * A)
print()
print(A * inv)
