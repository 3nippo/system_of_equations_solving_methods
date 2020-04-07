from matrix import Matrix, Norm
from matrix.utility_matrices import Householder
import math
import cmath
from collections import namedtuple
from equation.decomp import QRDecomp
from equation import Equation
from equation.iter_process import __IterProcess__


class EigenThings(__IterProcess__):
    __EigenData__ = namedtuple('EigenData', ['matrix', 'basis'])

    def __init__(self, A, error):
        A.assert_square()
        n = A.shape().rows

        assert n > 1, "Matrix order should be at least 2"

        super().__init__(error)

        self.init_x = A.copy()

    def rotation_method(self, norm=Norm.out_of_diagonal_norm):
        # unit matrix is required for storing eigenvectors
        A = self.init_x
        n = A.shape().rows

        self.init_x = EigenThings.__EigenData__(
            A,
            Matrix.unit_matrix(n)
        )

        answer = self.__iter_process__(
            EigenThings.__GetCurrent__.rotation_method,
            [],
            lambda last, _, norm: norm(last.matrix),
            [norm]
        )

        result_type = type(answer)
        (A, basis), iterations = answer

        eigenvalues  = [A[i][i] for i in range(n)]
        eigenvectors = [Matrix(n, 1, basis.get_column(i)) for i in range(n)]

        return result_type(
            (eigenvalues, eigenvectors),
            iterations
        )

    def qr_algorithm(self):
        save_init   = self.init_x
        self.init_x = Householder.to_hessenberg(self.init_x)

        answer = self.__iter_process__(
            EigenThings.__GetCurrent__.qr_algorithm,
            [QRDecomp],
            EigenThings.__GetDifference__.qr_algorithm,
            [self.error, [False] * min(*self.init_x.shape())]
        )

        self.init_x = save_init

        result_type = type(answer)

        A, iterations = answer

        eigenvalues = []

        continue_counter = 0
        n = min(*A.shape())
        for i in range(n):
            if continue_counter:
                continue_counter -= 1
                continue

            norm = EigenThings.__partial_R_2_norm__(A, i)

            if i < n - 1 and norm > self.error:
                continue_counter += 1
                eigenvalues.extend(EigenThings.__compute_eigenvals__(A, i))
            else:
                eigenvalues.append(A[i][i])

        return result_type(
            (eigenvalues, None),
            iterations
        )

    @staticmethod
    def __compute_eigenvals__(A, j):
        a = 1
        b = -(A[j][j] + A[j + 1][j + 1])
        c = -A[j][j + 1] * A[j + 1][j] + A[j][j] * A[j + 1][j + 1]

        sqrt_D = cmath.sqrt(b * b - 4 * a * c)

        x1 = (-b + sqrt_D) / 2 / a
        x2 = (-b - sqrt_D) / 2 / a

        return x1, x2

    @staticmethod
    def __cmp_computed__(a, b):
        dif1 = abs(a[0] - b[0])
        dif2 = abs(a[1] - b[1])

        return max(dif1, dif2)

    @staticmethod
    def __partial_R_2_norm__(A, i):
        norm = 0

        for j in range(i + 1, A.shape().rows):
            norm += A[j][i] * A[j][i]

        return math.sqrt(norm)

    class __GetDifference__:
        @staticmethod
        def qr_algorithm(last, curr, error, is_complex):
            n = min(*last.shape())

            counter = 0
            continue_counter = 0

            for i in range(n):
                if continue_counter:
                    continue_counter -= 1
                    continue

                if not is_complex[i]:
                    norm_curr = EigenThings.__partial_R_2_norm__(curr, i)

                    if abs(last[i][i] - curr[i][i]) <= error and \
                       norm_curr <= error:
                        continue

                    if i == n - 1:
                        counter += 1
                        continue

                before = EigenThings.__compute_eigenvals__(last, i)
                after  = EigenThings.__compute_eigenvals__(curr, i)

                if EigenThings.__cmp_computed__(before, after) > error or \
                   abs(after[0].imag) <= error:
                    counter += 1
                else:
                    is_complex[i] = True
                    continue_counter += 1

            return (error + 1) if counter > 0 else 0

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

        @staticmethod
        def qr_algorithm(last, c_decomp):
            """
            args:
                c_decomp --- class with QRDecomp interface
            """

            Q, R = c_decomp().decomp(last)

            return R * Q
