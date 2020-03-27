from iter_process import __IterProcess__
import math
import itertools
from matrix import Matrix, Norm


class Linear(__IterProcess__):
    def simple_iteration(self, A, B, norm=Norm.column_norm):
        return self.__iter_process__(
            Linear.__GetCurrent__.simple_iteration,
            *self.__calc_args_and_set_init_x__(A, B, norm)  # args_curr, diff func and args_diff
        )

    def zeydel_method(self, A, B, norm=Norm.column_norm):
        return self.__iter_process__(
            Linear.__GetCurrent__.zeydel,
            *self.__calc_args_and_set_init_x__(A, B, norm)  # args_curr, diff func and args_diff
        )

    def infimum_iterations_num(self, A, B, method='simple', norm=Norm.column_norm):
        Al, Bt = Linear.__preparation__(A, B)

        Al_norm = norm(Al)

        assert Al_norm < 1, "A do not satisfy sufficient condition"

        log = math.log
        divisor = log(Al_norm)

        if method == 'zeydel':
            upper_part = Al.copy()
            n = upper_part.shape().rows

            for i in range(n):
                for j in range(i):
                    upper_part[i][j] = 0

            divisor = log(norm(upper_part))

        return math.ceil((log(self.error) - log(norm(Bt)) + log(1 - Al_norm)) / divisor - 1)

    @staticmethod
    def __non_zero_diagonal__(A, B):
        m, n = A.shape()

        for perm in itertools.permutations(range(m)):
            failed = 0

            for i in range(n):
                if A[perm[i]][i] == 0:
                    failed = 1
                    break

            if not failed:
                break

        if failed:
            raise Linear.BadInputError

        return A.get_permutated(perm), B.get_permutated(perm)

    @staticmethod
    def __preparation__(A, B):
        m, n = A.shape()

        A, B = Linear.__non_zero_diagonal__(A, B)

        Al = Matrix(m, n)

        for i in range(m):
            for j in range(n):
                if i == j:
                    Al[i][j] = 0
                else:
                    Al[i][j] = -A[i][j] / A[i][i]

        Bt = Matrix(m, 1)

        for j in range(B.shape().columns):
            for i in range(m):
                Bt[i][j] = B[i][j] / A[i][i]

        return Al, Bt

    def __calc_args_and_set_init_x__(self, A, B, norm):
        Al, Bt = Linear.__preparation__(A, B)

        self.init_x = Bt

        Al_norm = norm(Al)
        error_const = Al_norm / (1 - Al_norm) if Al_norm < 1 else 1

        def difference(x_last, x_current, norm, error_const):
            return error_const * norm(x_last - x_current)

        return [Al, Bt], difference, [norm, error_const]

    # all funcs should take x_last first
    class __GetCurrent__:
        @staticmethod
        def simple_iteration(last, A, B):
            return B + A * last

        @staticmethod
        def zeydel(last, A, B):
            current = last.copy()

            n, k = A.shape().rows, B.shape().columns

            for i in range(k):
                for j in range(n):
                    current_cell = B[j][i]

                    for h in range(n):
                        current_cell += current[h][i] * A[j][h]

                    current[j][i] = current_cell

            return current

    class BadInputError(Exception):
        def __str__(self):
            return "Input matrix of equation is degenerate"