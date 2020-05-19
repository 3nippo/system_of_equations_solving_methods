from collections import namedtuple
import copy
import math


class __IApprox__:
    def __init__(self, X, Y):
        self.__coefs = []
        pass

    def get_coefs(self):
        return copy.copy(self.__coefs)

    def set_table(self, X, Y):
        pass

    def __call__(self, x):
        pass

    def n_plus_1_derivative(self):
        pass


class Lagrange(__IApprox__):
    def __init__(self, X=None, Y=None):
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
