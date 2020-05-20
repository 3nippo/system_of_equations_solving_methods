import context
from approx import Integral


def func(x):
    return x*x/(x*x + 16)


start = 0
end = 2
h = [0.5, 0.25]

methods = ['rectangle_method', 'trapeze_method', 'Simpson_method']
obj = Integral(start, end, func)

for step in h:
    obj.set_table(step)

    for method in methods:
        print(f"{method}, step = {step}")
        print(f"F = {getattr(obj, method)(step)}")
        print()

print("*"*15)
print("RungeRomberg_method")
print()


def sec_der(x):
    return (512-96*x**2)*(x**2+16)**(-3)


for method in methods:
    print(f"{method}")
    F, R = obj.RungeRomberg_method(method, h[0], h[1]/h[0])
    print(f"F = {F}")
    print(f"R = O({R})")
    print()
