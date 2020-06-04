import context
from approx import LeastSquares
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
        print()
        print(f"X = {X_row}")
        print("*"*15)
        print()

        f1 = method_class(X_row, Y_row, 1)
        f2 = method_class(X_row, Y_row, 2)

        print(f"Squared error for degree = 1: {f1.squared_error()}")
        print()
        print("Coefs:")
        f1.print_coefs()
        print("System of equations Ax=b")
        print("A")
        print(f1.A)
        print()
        print("b")
        print(f1.B)
        print()
        print(f"Squared error for degree = 2: {f2.squared_error()}")
        print()
        print("Coefs:")
        f2.print_coefs()
        print("System of equations Ax=b")
        print("A")
        print(f2.A)
        print()
        print("b")
        print(f2.B)
        print()

        more_X, Y1 = get_vals_to_draw(f1, X_row)
        _, Y2 = get_vals_to_draw(f2, X_row)

        plt.plot(more_X, Y1, label='degree 1')
        plt.plot(more_X, Y2, label='degree 2', linestyle='--')
        plt.plot(X_row, Y_row, label='target')
        plt.legend()
        plt.show()


X = [-5, -3, -1, 1, 3, 5]
Y = [
     -1.3734,
     -1.249,
     -0.7854,
     0.7854,
     1.249,
     1.3734
]

test_method(LeastSquares, [X], [Y])
