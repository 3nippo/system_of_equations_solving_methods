import context
from difeq import BoundaryVals
import math


def p(x):
    return -(2*x+1)/x


def q(x):
    return (x+1)/x


def f(x):
    return 0


def y(x):
    return x[2]


def z(x):
    return (2*x[0]+1)/x[0]*x[2] - (x[0]+1)/x[0]*x[1]


def f_true(x):
    return math.exp(x)*x*x


a = 1
b = 2
h = 0.1
eps = 0.00001


obj = BoundaryVals(a, b, h, f_true)

print(f"a = {a}, b = {b}, step = {h}, eps = {eps}")
print()


def difference(row):
    return abs(row['y'] - row['z'] / 2)


print("Shooting method")
print(obj.shooting_method(
    [0, 3*math.e],
    0,
    difference,
    eps,
    [y, z]
))
print()

obj.step = 0.05

print(f"step = {obj.step}")

print("Finite difference method")
print(obj.finite_difference(
    3*math.e,
    0,
    True,
    True,
    p,
    q,
    f,
    eps,
    b_coefs=[-1, 1-2*obj.step]
))

print("System of equations Ax=b")
print("A")
print(obj.A)
print()
print("b")
print(obj.B)
print()
