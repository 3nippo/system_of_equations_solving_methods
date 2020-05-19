import copy
from matrix import Matrix, TriDiagonalMatrix
from equation.decomp import LUDecomp


class Equation:
    class NotSuitableMethod(Exception):
        def __str__self(self):
            return "This method is not suitable for given matrix"

    def __init__(self, A=None, B=None, decomp=None):
        self.__LU = None

        if A is None:
            return

        self.__decomp = decomp or LUDecomp()
        self.__A = A.copy()
        self.__B = B.copy()

    def load_data(self, A, B, decomp=None):
        self.__decomp = decomp or LUDecomp()
        self.__A = A.copy()
        self.__LU = None
        self.__B = B.copy()
        return self

    def get_A(self):
        return self.__A.copy()

    def get_B(self):
        return copy.__B.copy()

    def get_LU(self):
        return self.__decomp.get_LU()

    def analytic_solution(self):
        decomp = self.__decomp

        if decomp.empty():
            decomp.decomp(self.__A, self.__B)

        LU, permuted_B = decomp.get_LU()

        if permuted_B is None:
            decomp.set_permuted_B(self.__B)
            permuted_B = decomp.get_LU()[1]

        m, _ = LU.shape()
        _, n = permuted_B.shape()
        X = Matrix(m, n)

        for k in range(n):
            y = [0] * m

            for i in range(m):
                l_sum = 0

                for j in range(i):
                    l_sum += y[j] * LU[i][j]

                y[i] = permuted_B[i][k] - l_sum

            for i in range(m - 1, -1, -1):
                l_sum = 0

                for j in range(i + 1, m):
                    l_sum += X[j][k] * LU[i][j]

                X[i][k] = (y[i] - l_sum) / LU[i][i]

        return X

    def sweep_method(self):
        A = self.__A
        B = self.__B

        m, _ = A.shape()
        _, n = B.shape()

        if not isinstance(A, TriDiagonalMatrix):
            raise Equation.NotSuitableMethod

        X = Matrix(m, n)

        for k in range(n):
            P = [0]
            Q = [0]

            for i in range(m - 1):
                a = A[i][0]
                b = A[i][1]
                c = A[i][2]
                d = B[i][k]

                denominator = b + a * P[-1]

                p = -c / denominator
                q = (d - a * Q[-1]) / denominator

                P.append(p)
                Q.append(q)

            a = A[m - 1][0]
            b = A[m - 1][1]
            # c = 0
            d = B[m - 1][k]

            q = (d - a * Q[-1]) / (b + a * P[-1])

            X[m - 1][k] = q

            for i in range(m - 2, -1, -1):
                X[i][k] = P[i + 1] * X[i + 1][k] + Q[i + 1]

        return X
