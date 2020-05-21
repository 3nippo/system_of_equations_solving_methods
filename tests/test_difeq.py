import context
from difeq import DifEq
import math


def y(x):
    return x[2]


def z(x):
    return -x[2]*math.tan(x[0]) - x[1]*math.cos(x[0])*math.cos(x[0])


def f_true(x):
    return math.cos(math.sin(x)) + math.sin(math.cos(x))


a = 0
b = 1
h = 0.1

obj = DifEq([0, 1], a, b, h, f_true)

print(f"a = {a}")
print(f"b = {b}")
print(f"h = {h}")
print()

print("Euler")
print(obj.Euler([y, z]))
print()
print("Runge Kutta")
print(obj.Runge_Kutta([y, z]))
print()
print("Adams")
print(obj.Adams([y, z], names=['y', 'z']))
