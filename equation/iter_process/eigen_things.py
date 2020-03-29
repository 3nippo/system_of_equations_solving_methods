from matrix import Matrix, Norm
from equation.iter_process import __IterProcess__
import math
from collections import namedtuple


class EigenThings(__IterProcess__):
    __Result__    = namedtuple('EigenThingsResult', ['eigenvalues', 'eigenvectors'])

    __EigenData__ = namedtuple('EigenData', ['matrix', 'basis'])

    def __init__(self, A, error):
        A.assert_square()
        n = A.shape().rows

        assert n > 1, "Matrix order should be at least 2"

        super().__init__(error)

        self.init_x = EigenThings.__EigenData__(A, Matrix(n))

    def rotation_method(self, norm=Norm.out_of_diagonal_norm):
        answer = self.__iter_process__(
            EigenThings.__GetCurrent__.rotation_method,
            [],
            lambda last, _, norm: norm(last.matrix),
            [norm]
        )

        result_type = type(answer)
        (A, basis), iterations = answer

        n, _ = A.shape()

        eigenvalues  = [A[i][i] for i in range(n)]
        eigenvectors = [Matrix(n, 1, basis.get_column(i)) for i in range(n)]

        return result_type(
            (eigenvalues, eigenvectors),
            iterations
        )

    class __GetCurrent__:
        @staticmethod
        def __abs_max__(A):
            '''
            Returns position of absolute maximum not from main diagonal
            '''
            h, w = 0, 1

            n, _ = A.shape()

            for i in range(n):
                for j in range(i + 1, n):
                    if abs(A[h][w]) < abs(A[i][j]):
                        h, w = i, j

            return h, w

        @staticmethod
        def rotation_method(last):
            A, basis = last

            i, j = EigenThings.__GetCurrent__.__abs_max__(A)

            if A[i][i] == A[j][j]:
                phi = math.pi / 4
            else:
                phi = math.atan(2 * A[i][j] / (A[i][i] - A[j][j])) / 2

            U = Matrix(*A.shape())

            U[i][i] = math.cos(phi)
            U[i][j] = -math.sin(phi)
            U[j][i] = math.sin(phi)
            U[j][j] = math.cos(phi)

            basis = basis * U
            A = U.transpose() * A * U

            return EigenThings.__EigenData__(A, basis)
