import sys
sys.path.append('../iter_process')
sys.path.append('../matrix')

from iter_process import NonLinear
import matplotlib.pyplot as plt
import math
from matrix import Matrix


def f1(x):
    return x[0] - math.cos(x[1]) - 1


def f2(x):
    return x[1] - math.sin(x[0]) - 1


def f1_reduced(x):
    return 1 + math.cos(x)


def f2_reduced(x):
    return math.asin(x - 1)


plt.xlabel('X_2')
plt.ylabel('X_1')


def plot(f, l, r, name):
    x2 = []
    x1 = []

    curr = l
    step = 0.05
    while curr <= r:
        x2.append(curr)
        x1.append(f(curr))
        curr += step

    plt.plot(x2, x1, label=name)


plot(f1_reduced, 0, 2, 'f1')  # as 0 <= sin(x1) = x2 - 1 <= 1
plot(f2_reduced, 0, 2, 'f2')

plt.grid(True)
plt.legend()
plt.title('f_1(x_1, x_2) = x_1 - Cos(x_2) - 1 = 0\nf_2(x_1, x_2) = x_2 - Sin(x_1) - 1 = 0')

plt.show()

# => G = { (x1, x2) : 0.6 <= x1 <= 1, 1.6 <= x2 <= 2 }
#
#           |    0      -Sin(x2) | , let norm be R_infinity (||x|| = max|xi| where i from N)
# phi'(x) = | Cos(x1)      0     | , then max ||phi'(x)|| <= max(|Cos(x1)|, |-Sin(x2)|) = max(Cos(0.6), Sin(1.6)) < 1


phi_vec = Matrix(2, 1, [lambda x: 1 + math.cos(x[1]),
                        lambda x: 1 + math.sin(x[0])])

func_vec = Matrix(2, 1, [f1, f2])

Jacobi_matrix = Matrix(2, 2, [       lambda x: 1,        lambda x: math.sin(x[1]),
                              lambda x: -math.cos(x[0]),       lambda x: 1        ])


def some_work(name, x_init, a_error, method_name, method_args):

    print("### {} ###".format(name))

    print("Given error: {}".format(a_error))

    print(
        "Answer:",
        getattr(NonLinear(x_init, a_error), method_name)(*method_args).answer
    )

    print()

    y = [2**i for i in range(-5, -25, -1)]

    x = [getattr(NonLinear(x_init, error), method_name)(*method_args).iterations for error in y]

    plt.plot(x, y)
    plt.xlabel('iterations')
    plt.ylabel('error')
    plt.title('{}\nError -- Number of iterations\ndependence'.format(name))
    plt.show()


init_x = Matrix(2, 1, [0.6, 1.6])

some_work(
    'Simple iterations',
    init_x,
    2**(-10),
    'simple_iteration',
    [phi_vec]
)

some_work(
    'Newton method',
    init_x,
    2**(-10),
    'Newton_method',
    [func_vec, Jacobi_matrix]
)
