import context
from matrix import Matrix, TriDiagonalMatrix
from equation import Equation

elems = [ 0, -7, -6,
          6, 12,  0,
         -3,  5,  0,
         -9, 21,  8,
         -5, -6,  0]

triA = TriDiagonalMatrix(5, elems)

elems = [-75, 126, 13, -40, -24]

B = Matrix(5, 1, elems)

tri_equation = Equation(triA, B)
tri_solution = tri_equation.sweep_method()

print("Solution:")
print(tri_solution)
print()
print("Given:")
print(B)
print()
print("Calculated:")
print(triA * tri_solution)
