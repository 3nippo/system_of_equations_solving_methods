import context
from approx import CubicSpline
import matplotlib.pyplot as plt


def get_vals_to_draw(f, X, step=0.1):
    step = 0.1
    pos = min(X)
    end = max(X)

    aX = []
    aY = []
    while pos <= end:
        aX.append(pos)
        aY.append(f(pos))
        pos += step

    return aX, aY


def test_method(method_class, X, Y):
    print(method_class.__name__)

    for X_row, Y_row in zip(X, Y):
        print(f"X = {X_row}")
        print("*"*15)
        print()

        f = method_class(X_row, Y_row)

        f.print_coefs()
        print()

        print("System of equations Ax=b")
        print("A")
        print(f.triA.to_Matrix())
        print()
        print("b")
        print(f.B)
        print()

        more_X, approx_Y = get_vals_to_draw(f, X_row)

        plt.plot(more_X, approx_Y)
        plt.show()


X = [-3, -1, 1, 3, 5]
Y = [
     -1.2490,
     -0.78540,
     0.78540,
     1.2490,
     1.3734
]

to_calc = -0.5

test_method(CubicSpline, [X], [Y])

print(f"S({-0.5}) = {CubicSpline(X, Y)(-0.5)}")
