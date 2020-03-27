from matrix import Matrix, Norm
import itertools
import math
from collections import namedtuple


class __IterProcess__:
    __Result__ = namedtuple('IterProcessResult', ['answer', 'iterations'])

    def __init__(self, error):
        self.error  = error
        self.init_x = None  # will be provided

    def __iter_process__(self, current, args_curr, difference, args_diff):
        if not self.init_x:
            raise NotImplementedError

        x_last = self.init_x
        x_current = current(x_last, *args_curr)

        iterations = 1

        while difference(x_last, x_current, *args_diff) > self.error:
            x_last    = x_current
            x_current = current(x_last, *args_curr)

            iterations += 1

        return __IterProcess__.__Result__(x_current, iterations)


class NonLinear(__IterProcess__):
    def __init__(self, init_x, error):
        super().__init__(error)
        self.init_x = init_x

    def simple_iteration(self, reduced_func_vec, norm=Norm.R_infinity_norm):
        return self.__iter_process__(
            NonLinear.__GetCurrent__.simple_iteration,
            [reduced_func_vec],
            lambda x_last, x_current, l_norm: l_norm(x_last - x_current),
            [norm]
        )

    def Newton_method(self, func_vec, Jacobi_matrix, norm=Norm.R_infinity_norm):
        return self.__iter_process__(
            NonLinear.__GetCurrent__.Newton,
            [self.error, func_vec, Jacobi_matrix],
            lambda x_last, x_current, l_norm: l_norm(x_last - x_current),
            [norm]
        )

    # all funcs should take x_last first
    class __GetCurrent__:
        @staticmethod
        def Newton(last, error, func_vec, Jacobi_matrix):
            right_part = -func_vec(last)

            return Linear(error).zeydel_method(  # give right part as init value
                Jacobi_matrix(last),
                right_part
            )[0] + last

        @staticmethod
        def simple_iteration(last, reduced_func_vec):
            return reduced_func_vec(last)


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
