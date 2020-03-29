import context
from equation.iter_process.non_linear import NonLinear
import matplotlib.pyplot as plt
import math
from matrix import Matrix


def f(x):
    return math.sin(x[0]) - 2 * x[0] * x[0] + 0.5


x = []
y = []

curr = -1
step = 0.05
while curr <= 1:
    x.append(curr)
    y.append(f([curr]))
    curr += step

plt.plot(x, y)
plt.plot([-1, 1], [0, 0])
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.title('y(x) = Sin(x) - 2*x^2 + 0.5')

plt.show()


def simple_iterations_function(x):  # it is phi function
    return math.sqrt(2 * math.sin(x[0]) + 1) / 2


def first_derivative(x):
    return math.cos(x[0]) - 4 * x[0]


def second_derivative(x):
    return -math.sin(x[0]) - 4


class Newton:
    def __init__(self, f, f_der, s_der, init_x):
        assert f([init_x]) * s_der([init_x]) > 0, "Newton condition is not satisfied"

        self.__f_der = f_der
        self.__f = f

    def get_args(self):
        return Matrix(1, 1, [self.__f]), Matrix(1, 1, [self.__f_der])


def some_work(name, x_init, a_error, method_name, method_args):

    print("### {} ###".format(name))

    print("Given error: {}".format(a_error))
    print("Initial value: {}".format(x_init))

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


simple_func_vec = Matrix(1, 1, [simple_iterations_function])

init_x = Matrix(1, 1, [math.pi])

some_work(
    'Simple iterations',
    init_x,
    2**(-10),
    'simple_iteration',
    [simple_func_vec]
)

some_work(
    'Newton method',
    init_x,
    2**(-10),
    'Newton_method',
    Newton(f, first_derivative, second_derivative, math.pi).get_args()
)
