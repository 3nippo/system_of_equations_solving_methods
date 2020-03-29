import context
from matrix import Matrix
from equation import Equation
from equation.iter_process.linear import Linear

A = Matrix(4, elems = [28,  9,  -3, -7,
                       -5, 21,  -5, -3,
                       -8,  1, -16,  5,
                        0, -2,   5,  8]
)

B = Matrix(1, 4, [-159, 63, -45, 24])
B = B.transpose()

eq = Equation(A, B)

X_analytic = eq.analytic_solution()

error = 0.000001
lin_iter = Linear(error)

X_simple, simple_actual_iterations = lin_iter.simple_iteration(A, B)
X_zeydel, zeydel_actual_iterations = lin_iter.zeydel_method(A, B)

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
print("--Infimum:", lin_iter.infimum_iterations_num(A, B))
print("--Actual :", simple_actual_iterations)
print()

print("Zeydel meothod number of iterations")
print("--Infimum:", lin_iter.infimum_iterations_num(A, B, method='zeydel'))
print("--Actual :", zeydel_actual_iterations)
print()

print("Indeed, formula gives overstated number of iterations")
