import context
import approx
import math
import matplotlib.pyplot as plt

X = [
     [-3, -1, 1, 3],
     [-3, 0, 1, 3]
]

Y = [[math.atan(el) for el in row] for row in X]

x_for_estimation = -0.5


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


def test_method(method_class, X, Y, func):
    print(method.__name__)

    for X_row, Y_row in zip(X, Y):
        print()
        print(f"X = {X_row}")
        print("*"*15)
        print()

        f = method_class(X_row, Y_row)

        for x, y in zip(X_row, Y_row):
            print(f"x = {x}")
            print(f"Actual value: {func(x)}")
            print(f"Approxed value: {f(x)}")
            print()

        more_X, more_Y = get_vals_to_draw(func, X_row)
        _, approx_Y = get_vals_to_draw(f, X_row)

        plt.plot(more_X, more_Y, label='Actual')
        plt.plot(more_X, approx_Y, label='Approxed')
        plt.legend()
        plt.show()


methods = [approx.Lagrange, approx.Newton]

for method in methods:
    test_method(method, X, Y, math.atan)


def n_plus_1_derivative(x):
    return 24*x*(1-x*x)*(x*x+1)**(-4)


more_X, more_Y = get_vals_to_draw(n_plus_1_derivative, X[0], 0.01)

plt.plot(more_X, more_Y)
plt.title('4th derivative')
plt.show()

more_Y = list(map(abs, more_Y))

max_der = max(more_Y)
fact = math.factorial(len(X[0]))

print(f"Error estimation for x = {x_for_estimation}")

for XX in X:
    print("*"*15)
    print(f"Interpolation based on X = {XX}")

    omega = 1

    for x in XX:
        omega *= (x_for_estimation - x)

    print(f"Error <= {abs(omega) * max_der / fact}")
