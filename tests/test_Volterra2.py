import context
import difeq
import math


a = 0
b = 0.1
h = 0.01
N = int((b - a) / h)
epst = 0.001


def A_Simpson(i):
    if i == 0 or i == N:
        return h/3

    if i % 2:
        return 4*h/3

    return 2*h/3


def get_A(i):
    def A(j):
        if j == i:
            return h/2
        return h
    return A


def K(x1, x2, y):
    return math.exp(-(x1 - x2)) * y * y


def f(x):
    return math.exp(-x)


solver = difeq.Volterra2(
    K,
    f,
    get_A(0),
    a,
    b,
    h
)


def y(i, c):
    y2 = (1 - math.sqrt(1 - 2*h*c)) / h
    return y2


assert f(a) == solver.y[0][0]

all_y = []

for i in range(1, N+1):
    solver.A = get_A(i)

    _, x, Ai, ci = solver.next_equation()

    ci = ci[0]

    yi = y(i, ci)
    all_y.append(yi)

    def func_vec(y):
        return y[0] - Ai*K(x, x, y[0]) - ci

    solver.solve_current(
        ['dichotomy'],
        [[(yi - epst, yi + epst), func_vec]]
    )

my_y = solver.get_solutions()[0]

all_y = [f(a)] + all_y

for i, (y_true, y_mine) in enumerate(zip(all_y, my_y)):
    print(f"Value #{i}")
    print(f"given: {y_true}")
    print(f"calculated: {y_mine}")
    print()
