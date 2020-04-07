from matrix import Matrix
import math


class __IDecomposition__:
    class CalledBeforeDecompError(Exception):
        def __str__(self):
            return "Decomp matrix has not been constructed yet"

    class EmptyMatrixError(Exception):
        def __str__(self):
            return "Given empty matrix of shape (0, 0)"

        def empty(self):
            pass

        def decomp(self, *args, **kwargs):
            pass


class LUDecomp(__IDecomposition__):
    def __init__(self, A=None, B=None):
        if A is None:
            self.__LU = None
            return
        self.decomp(A, B)

    def get_LU(self):
        if self.empty():
            raise __IDecomposition__.CalledBeforeDecompError

        return self.__LU, self.__B

    def empty(self):
        return self.__LU is None

    def decomp(self, A, B=None):
        if A.shape().rows < 1:
            raise __IDecomposition__.EmptyMatrixError

        self.__A = A.copy()
        B = self.__B = B.copy() if B else None

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
            raise __IDecomposition__.CalledBeforeDecompError

        det = 1
        LU = self.__LU

        for i in range(LU.shape().rows):
            det *= LU[i][i]

        return det * (-1 if self.__perms_count % 2 else 1)

    def inverse_matrix(self, equation):
        """
            args:
                equation - class with Equation class interface
        """
        if self.empty():
            raise __IDecomposition__.CalledBeforeDecompError

        A = self.__A

        previous_B = self.__B

        new_B = Matrix.unit_matrix(A.shape().rows)
        self.set_permuted_B(new_B)

        result = equation(A, new_B, self).analytic_solution()

        self.__B = previous_B

        return result


class QRDecomp(__IDecomposition__):
    @staticmethod
    def __Householder_matrix__(V):
        m, n = V.shape()

        assert n == 1, "V should be vector"

        VT = V.transpose()

        scalar_product = (VT * V).to_val()
        matrix = V * VT

        return Matrix.unit_matrix(m) - 2 / scalar_product * matrix

    def empty(self):
        return 'Q' not in self.__dict__

    def get_QR(self):
        if self.empty():
            raise __IDecomposition__.CalledBeforeDecompError

        return self.Q, self.R

    def decomp(self, A):
        R = A.copy()

        m, n = A.shape()

        Q = Matrix.unit_matrix(m)

        for i in range(min(m, n)):
            vec_elems = [0 for _ in range(i)]

            norm = 0

            for j in range(i, m):
                norm += R[j][i] * R[j][i]

            norm = math.sqrt(norm)

            vec_elems.append(
                R[i][i] + norm * (-1 if R[i][i] < 0 else 1)
            )

            for j in range(i + 1, m):
                vec_elems.append(R[j][i])

            H = QRDecomp.__Householder_matrix__(
                Matrix(m, 1, vec_elems)
            )

            Q = Q * H
            R = H * R

        self.Q = Q
        self.R = R

        return Q, R
