import context
from matrix import Matrix
from equation import LUDecomposition

A = [8, 1, 1,
     5, 3, 4,
     7, 2, 7]

A = Matrix(3, elems=A)
lu = LUDecomposition(A)

print(A)
print()
print(lu.det())
print()
print(lu.inverse_matrix())
print()
print(A * lu.inverse_matrix())
print()
print(lu.inverse_matrix() * A)

A = Matrix()
LUDecomposition(A)  # should throw an exception as empty matrix was given
