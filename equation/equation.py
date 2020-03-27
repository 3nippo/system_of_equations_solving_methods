import copy
from matrix import Matrix, TriDiagonalMatrix


class LUDecomposition:
    class CalledBeforeLUError(Exception):
        def __str__(self):
            return "LU matrix has not been constructed yet"

    class EmptyMatrixError(Exception):
        def __str__(self):
            return "Given empty matrix of shape (0, 0)"

    def __init__(self, A=None, B=None):
        if A is None:
            self.__LU = None
            return
        self.decomp(A, B)

    def get_LU(self):
        if self.empty():
            raise LUDecomposition.CalledBeforeLUError

        if self.__B:
            return self.__LU, self.__B
        return self.__LU, None

    def empty(self):
        return self.__LU is None

    def decomp(self, A, _B=None):
        if A.shape().rows == 0:
            raise LUDecomposition.EmptyMatrixError

        self.__A = A.copy()
        B = self.__B = _B.copy() if _B else None

        n, _ = A.shape()
        LU = self.__LU = A.copy()

        # save permutations for possible usage
        perms = [(i, i) for i in range(n)]
        perms = self.__perms = dict(perms)
        perms_count = 0

        for k in range(n):
            max_row = LU.get_index_abs_max(k, k, 1)

            if LU[max_row][k] > LU[k][k]:
                perms_count += 1

                LU.swap(k, max_row, 0)
                perms[k], perms[max_row] = perms[max_row], perms[k]

                if B:
                    B.swap(k, max_row, 0)

            for i in range(k + 1, n):
                mu = LU[i][k] / LU[k][k]

                for j in range(k, n):
                    LU[i][j] -= mu * LU[k][j]

                LU[i][k] = mu

        self.__perms_count = perms_count

        return self.get_LU()

    def set_permuted_B(self, B):
        m, n = B.shape()
        res = self.__B = Matrix(m, n)

        for i in range(m):
            res[i] = B[self.__perms[i]]

    # determinant
    def det(self):
        if self.empty():
            raise LUDecomposition.CalledBeforeLUError

        det = 1
        LU = self.__LU

        for i in range(LU.shape().rows):
            det *= LU[i][i]

        return det * (-1 if self.__perms_count % 2 else 1)

    def inverse_matrix(self, equation=None):
        if self.empty():
            raise LUDecomposition.CalledBeforeLUError

        A = self.__A

        equation = equation or Equation()

        previous_B = self.__B

        new_B = Matrix.unit_matrix(A.shape().rows)
        self.set_permuted_B(new_B)

        equation.load_data(A, new_B, self)
        result = equation.analytic_solution()

        self.__B = previous_B

        return result


class Equation:
    class NotSuitableMethod(Exception):
        def __str__self(self):
            return "This method is not suitable for given matrix"

    def __init__(self, A=None, B=None, decomp=None):
        self.__LU = None

        if A is None:
            return

        self.__decomp = decomp or LUDecomposition()
        self.__A = A.copy()
        self.__B = B.copy()

    def load_data(self, A, B, decomp=None):
        self.__decomp = decomp or LUDecomposition()
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
            a = 0
            b = A[0][1]
            c = A[0][2]
            d = B[0][k]

            P = [-c / b]
            Q = [ d / b]

            for i in range(1, m - 1):
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
                X[i][k] = P[i] * X[i + 1][k] + Q[i]

        return X
