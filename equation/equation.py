import copy
from matrix import Matrix


class CalledBeforeLUError(Exception):
    def __str__(self):
        return "LU matrix has not been constructed yet"


class EmptyMatrixError(Exception):
    def __str__(self):
        return "Given empty matrix of shape (0, 0)"


class LUDecomposition:
    def __init__(self, A=None, B=None):
        if A is None:
            self.__LU = None
            return
        self.decomp(A, B)

    def get_LU(self):
        if self.empty():
            raise CalledBeforeLUError

        if self.__B:
            return self.__LU, self.__B
        return self.__LU, None

    def empty(self):
        return self.__LU is None

    def decomp(self, A, _B=None):
        if A.shape()[0] == 0:
            raise EmptyMatrixError

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
            raise CalledBeforeLUError

        det = 1
        LU = self.__LU

        for i in range(LU.shape()[0]):
            det *= LU[i][i]

        return det * (-1 if self.__perms_count % 2 else 1)

    def inverse_matrix(self, equation=None):
        if self.empty():
            raise CalledBeforeLUError

        A = self.__A

        equation = equation or Equation()

        previous_B = self.__B

        new_B = Matrix.unit_matrix(A.shape()[0])
        self.set_permuted_B(new_B)

        equation.load_data(A, new_B, self)
        result = equation.analytic_solution()

        self.__B = previous_B

        return result


class Equation:
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
