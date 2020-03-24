import sys
sys.path.append('../matrix')
sys.path.append('../equation')

from matrix import Matrix
from equation import Equation

A = Matrix(4, elems = [28,  9,  -3, -7,
                       -5, 21,  -5, -3,
                       -8,  1, -16,  5,
                        0, -2,   5,  8]
)

B = Matrix(1, 4, [-159, 63, -45, 24])
B = B.transpose()

eq = Equation(A, B)
error = 0.000001

X_analytic = eq.analytic_solution()
X_simple   = eq.simple_iterations(error)
X_zeydel   = eq.zeydel_method(error)

print("Analytic answer:")
print(X_analytic)
print()

print("Calculated:")
print(A * X_analytic)
print()

print("Given:")
print(B)
print()

print("Simple iterations answer:")
print(X_simple)
print()

print("Zeydel iterations answer:")
print(X_zeydel)
print()

print("Simple iterations number of iterations")
print(eq.infimum_iterations_num(error, Matrix.column_norm))
print()

print("Zeydel meothod number of iterations")
print(eq.infimum_iterations_num(error, Matrix.column_norm, 'zeydel'))
