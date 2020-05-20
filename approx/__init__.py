from collections import namedtuple
import copy
import math
from matrix import Matrix, TriDiagonalMatrix
from equation import Equation
from bisect import bisect_left


class __IApprox__:
    def __init__(self, X=None, Y=None):
        self.__coefs = []
        pass

    def get_coefs(self):
        return copy.copy(self.__coefs)

    def set_table(self, X, Y, *args):
        pass

    def __call__(self, x):
        pass

    def n_plus_1_derivative(self):
        pass


class Lagrange(__IApprox__):
    def __init__(self, X=None, Y=None):
        super().__init__()
        if X:
            self.set_table(X, Y)

    def set_table(self, X, Y):
        self.__X = X.copy()
        self.__Y = Y.copy()

    def n_plus_1_derivative(self):
        coefs = self.__coefs
        return sum(coefs) * math.factorial(len(coefs))

    def __call__(self, x):
        sum = 0
        X = self.__X
        Y = self.__Y
        coefs = self.__coefs = []

        for i in range(len(X)):
            mult = 1
            c = 1

            for j in range(len(X)):
                if i == j:
                    continue
                c /= (X[i] - X[j])
                mult *= (x - X[j]) / (X[i] - X[j])

            sum += mult * Y[i]
            coefs.append(c * Y[i])

        return sum


class Newton(__IApprox__):
    """
    last_div_diff --- last divided difference
    """
    class __DivDiff__:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __repr__(self):
            return f"{self.x} {self.y}"

        def __copy__(self):
            return Newton.__DivDiff__(self.x, self.y)

    def __init__(self, X=None, Y=None):
        super().__init__()
        if X:
            self.set_table(X, Y)

    def n_plus_1_derivative(self):
        coefs = self.__coefs
        return coefs[-1] * math.factorial(len(coefs))

    def clear(self):
        self.__X = []
        self.__Y = []
        self.__last_div_diff = []
        self.__coefs = []

    def set_table(self, X, Y):
        self.clear()

        for x, y in zip(X, Y):
            self.add(x, y)

    def add(self, x, y):
        self.__X.append(x)
        self.__Y.append(y)

        last_div_diff = self.__last_div_diff

        next = Newton.__DivDiff__(x, y)

        for i in range(len(last_div_diff)):
            current = last_div_diff[i]

            last_div_diff[i] = copy.copy(next)

            next.y = (current.y - next.y) / (current.x - x)
            next.x = current.x

        last_div_diff.append(next)

        self.__coefs.append(next.y)

    def __call__(self, x):
        f = self.__coefs

        sum = f[0]
        mult = 1

        for i in range(1, len(f)):
            mult *= (x - self.__X[i - 1])
            sum += f[i] * mult

        return sum


class CubicSpline(__IApprox__):
    def __init__(self, X=None, Y=None):
        super().__init__()
        if X:
            self.set_table(X, Y)

    def set_table(self, X, Y):
        X = copy.copy(X)
        Y = copy.copy(Y)

        l = list(zip(X, Y))

        l = sorted(l, key=lambda x: x[0])

        X, Y = map(list, zip(*l))

        self.__X = X
        self.__Y = Y

        self.__build_func__()

    def __build_func__(self):
        X = self.__X
        Y = self.__Y

        h = [X[i] - X[i-1] for i in range(1, len(X))]

        A_elems = [2 * (h[0] + h[1]), h[1]]
        B_elems = [3 * ((Y[2] - Y[1]) / h[1] - (Y[1] - Y[0]) / h[0])]

        for i in range(2, len(h) - 1):
            A_elems.append(h[i-1])
            A_elems.append(2 * (h[i-1] + h[i]))
            A_elems.append(h[i])

            B_elems.append(3 * ((Y[i+1] - Y[i]) / h[i] - (Y[i] - Y[i-1]) / h[i-1]))

        A_elems.append(h[-2])
        A_elems.append(2 * (h[-2] + h[-1]))

        B_elems.append(3 * ((Y[-1] - Y[-2]) / h[-1] - (Y[-2] - Y[-3]) / h[-2]))

        triA = TriDiagonalMatrix(len(h) - 1, A_elems)
        B = Matrix(len(h) - 1, 1, B_elems)

        tri_equation = Equation(triA, B)
        tri_solution = tri_equation.sweep_method()

        a = Y[:-1]

        c = [0]
        c.extend(tri_solution.to_list())

        b = []
        d = []
        for i in range(len(h) - 1):
            b.append((Y[i+1] - Y[i]) / h[i] - 1 / 3 * h[i] * (c[i+1] + 2 * c[i]))

            d.append((c[i+1] - c[i]) / 3 / h[i])

        b.append((Y[-1] - Y[-2]) / h[-1] - 2 / 3 * h[-1] * c[-1])

        d.append(-c[-1] / 3 / h[-1])

        self.__a = a
        self.__b = b
        self.__c = c
        self.__d = d

    def __call__(self, x):
        a = self.__a
        b = self.__b
        c = self.__c
        d = self.__d
        X = self.__X

        if x > X[-1]:
            i = len(a) - 1
        elif x <= X[0]:
            i = 0
        else:
            i = bisect_left(X, x) - 1

        return a[i] + b[i] * (x - X[i]) + c[i] * (x - X[i])**2 + d[i] * (x - X[i])**3


class LeastSquares(__IApprox__):
    def __init__(self, X=None, Y=None, n=1):
        super().__init__()
        self.set_table(X, Y, n)

    def set_table(self, X, Y, n):
        self.__X = copy.copy(X)
        self.__Y = copy.copy(Y)
        self.__n = n

        A_elems = []
        B_elems = []

        for k in range(n+1):
            for i in range(n+1):
                coef = 0

                for x in X:
                    coef += x**(k+i)

                A_elems.append(coef)

            b = 0

            for x, y in zip(X, Y):
                b += y * x**k

            B_elems.append(b)

        A = Matrix(n+1, elems=A_elems)
        B = Matrix(n+1, 1, B_elems)

        eq = Equation(A, B)

        self.__a = eq.analytic_solution().to_list()

    def __call__(self, x):
        sum = 0

        for i in range(self.__n+1):
            sum += self.__a[i] * x**i

        return sum

    def squared_error(self):
        sum = 0

        for x, y in zip(self.__X, self.__Y):
            sum += (self(x) - y)**2

        return sum


def choose_point(X, x):
    if x > X[-1]:
        return len(X) - 1, len(X) - 1

    if x < X[0]:
        return 0, 0

    i = bisect_left(X, x)

    if x == X[i]:
        return i

    return i - 1, i


def first_derivative(X, Y, x):
    i = choose_point(X, x)

    if type(i) == tuple:
        l, r = i
        accuracy_1 = (Y[r] - Y[l]) / (X[r] - X[l])

        if r + 1 == len(X):
            accuracy_2 = None
        else:
            numerator = (Y[r+1] - Y[r])/(X[r+1] - X[r]) - (Y[r] - Y[l])/(X[r] - X[l])
            accuracy_2 = (Y[i+1] - Y[i])/(X[i+1]-X[i]) + numerator / (X[r + 1] - X[l])*(2*x-X[i]-X[i+1])
    else:
        if i - 1 == -1:
            from_left = None
        else:
            from_left = (Y[i] - Y[i - 1]) / (X[i] - X[i - 1])

        if i + 1 == len(X):
            from_right = None
        else:
            from_right = (Y[i + 1] - Y[i]) / (X[i + 1] - X[i])

        accuracy_1 = from_left, from_right

        if not (from_left or from_right):
            accuracy_2 = None
        else:
            numerator = (Y[i+1] - Y[i])/(X[i+1] - X[i]) - (Y[i] - Y[i-1])/(X[i] - X[i-1])
            accuracy_2 = (Y[i+1] - Y[i])/(X[i+1]-X[i]) + numerator / (X[i + 1] - X[i - 1])*(2*x-X[i]-X[i+1])

    return accuracy_1, accuracy_2


def second_derivative(X, Y, x):
    i = choose_point(X, x)

    if type(i) == tuple:
        l, r = i

        if r + 1 == len(X):
            answer = None
        else:
            numerator = (Y[r+1] - Y[r])/(X[r+1] - X[r]) - (Y[r] - Y[l])/(X[r] - X[l])
            answer = 2 * numerator / (X[r + 1] - X[l])
    else:
        if i - 1 == -1 or i + 1 == len(X):
            answer = None
        else:
            numerator = (Y[i+1] - Y[i])/(X[i+1] - X[i]) - (Y[i] - Y[i-1])/(X[i] - X[i-1])
            answer = 2 * numerator / (X[i + 1] - X[i - 1])

    return answer
